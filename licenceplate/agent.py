import cv2, numpy, json, logging, logging.config, openalpr, threading
import errno, hashlib, random, string, time, uuid, zipfile
from datetime import datetime
from functools import partial
from glob import glob
from os import makedirs, path, rename, remove
from queue import Queue
from . import processing
from licenceplate.centroid_tracker import CentroidTracker


class Agent:

    STATUS_STOPPED = 0
    STATUS_CAMERA_RUNNING = 2

    def __init__(self, tag_slug):
        logging.config.fileConfig(path.join(path.dirname(path.abspath(__file__)), 'config', 'logging.conf'))

        self.tag_slug = tag_slug
        self.logger = logging.getLogger('agent')
        
        self._local_storage_lock = threading.Lock()
        self._local_storage_dir = path.join('plugins-storage', 'licenceplate', 'local', tag_slug)
        try:
            makedirs(self._local_storage_dir)
        except OSError as error:
            if error.errno != errno.EEXIST: raise

        self._final_storage_dir = path.join('plugins-storage', 'licenceplate', 'final', tag_slug)
        try:
            makedirs(self._final_storage_dir)
        except OSError as error:
            if error.errno != errno.EEXIST: raise

        self._processing_rate_lock = threading.Lock()
        self._processing_counter = 0
        self._processing_counter_start = None
        self._processing_rate = 0.
        
        self._stream = None
        self._keep_running = False
        
        self.logger.info(f"{self.tag_slug}\tinit")

        # Cria o rastreamaneto
        self.tracking = CentroidTracker()

        # Guarda o ID e e matriz da placa do veiculo rastreado
        self.listaPlacas = list()

    def __del__(self):
        self.logger.info(f"{self.tag_slug}\tdel")

    def _run_frame_worker(self, index, frame_queue, storage_queue, frame_callback):
        self.logger.debug('%s\tframe worker #%d - begin' % (self.tag_slug, index))
        try:
            # inicia a identificação das placas
            alpr = openalpr.Alpr('br', path.join(path.dirname(path.abspath(__file__)), 'config', 'openalpr.conf'), '')
            if not alpr.is_loaded(): raise Exception('Error loading OpenALPR')
            try:
                # Define quantos canditados serão usados
                alpr.set_top_n(10) 
                frame, when = frame_queue.get()
                confidence = None
                candidates = None
                roi = None
                while frame is not None:
                    
                    # Vetor que guarda os centroids e passa para o tracking
                    centroidList = []
                    # Dicionário para guardar centroid e placa
                    ocrs = dict()  
                    if frame_callback is not None: frame_callback(self, {'frame': frame, 'when': when})
                    for result in alpr.recognize_ndarray(frame)['results']:
                        coords = result['coordinates']
                        confidence = result['confidence']
                        candidates = [{'plate': candidate['plate'], 'confidence': candidate['confidence']} for candidate in result['candidates']]

                        x, y = [point['x'] for point in coords], [point['y'] for point in coords]
                        roi = frame[numpy.amin(y):numpy.amax(y), numpy.amin(x):numpy.amax(x), :]
                        # Calcula o centroid do ROI
                        cX = int((int(x[0]) + int(x[2])) / 2.0)
                        cY = int((int(y[0]) + int(y[2])) / 2.0)
                        centroidALPR = (cX, cY)                        
                        # Guarda na lista o centroid 
                        centroidList.append(centroidALPR)
                        # Guarda a placa que relacionada aquele centroid
                        ocrs[centroidALPR] = result['plate']
                        
                        ####################################################                     
                        
                        ###################################################
                    # Recebe um vetor com as placas
                    placas = self._tracking(ocrs, centroidList, frame)
                    # Se existe pelo menos uma placa em placas
                    if len(placas) > 0:                        
                        for placa in placas:
                            placaFinal = placa[2]
                            frameFinal = placa[1]
                            melhorPlaca = ''
                            # Para cada caracter na placa pecorre o vetor
                            # referente a ele e pega o caracter com o
                            # maior valor (aquele que mais se repetiu ao longo do 
                            # rastreamento) 
                            for j in range(7):
                                max_value = max(placaFinal[j])
                                indice = placaFinal[j].index(max_value)
                                caracter = chr(indice + ord("0"))
                                melhorPlaca += caracter
                                pass
                            # Salva na na fila storage_queue a placa após ela deixar de rastreada
                            # Esta técnica impede que a mesma placa seja salva varias vezes
                            storage_queue.put({
                                'info': {
                                    'plate': melhorPlaca,
                                    'confidence': confidence,
                                    'candidates': candidates,
                                    'when': when,
                                    'camera': {
                                        'tag_slug': self.tag_slug,
                                    },
                                },
                                'frame': frameFinal,
                                'roi': roi,
                            })
                            self.logger.debug('\tPlaca a ser salva no banco: %s' % (melhorPlaca))
                    
                    with self._processing_rate_lock:
                        self._processing_counter += 1
                    frame_queue.task_done()
                    frame, when = frame_queue.get()
                frame_queue.task_done()
            finally:
                alpr.unload()
        except:
            self.logger.exception('%s\tframe worker #%d - end by exception' % (self.tag_slug, index))
            raise            
        self.logger.debug('%s\tframe worker #%d - end' % (self.tag_slug, index))
        
    def _tracking(self, ocrs, centroidList, frame):
        # Guarda o id e placa capturadas em um frame
        idPlaca = dict()
        # Lista com placas para serem salvas no BD
        placaFinal = list()
        # dicionário para relacionar o id e o frame
        idFrame = dict()
        # Recebe  vetor com os centroids identificador pelo OpenALPR e retorna o
        # Dicionário contendo o id e centroid dos objetos rastreados
        objectsTracking = self.tracking.update(centroidList)
        # Associa a placa ao ID
        for (centroidOpenALPR, placa) in ocrs.items():
            for (objectID, centroidTracking) in objectsTracking.items():      
                centroidConvertido = (centroidTracking[0], centroidTracking[1])            
                if centroidOpenALPR == centroidConvertido: 
                    idPlaca[objectID] = placa
                    idFrame[objectID] = frame
        # Verificar se a placa já esta sendo rastreada
        for (id, placa) in idPlaca.items():
            # Matriz para armazenar quantas vezes os caracteres
            # da mesma placa se repete ao longo do rastreamento
            arrayPlaca = [[0] * 43 for i in range(7)]
            # Cada vez que o caracter se repete naquela posição incrementa 1
            # na matriz
            for i in range(7):
                ind = ord(placa[i]) - ord("0")
                arrayPlaca[i][ind] += 1
            encontrou = False
            # Se não houver placas em listaPlaca apenas adiciona
            if len(self.listaPlacas) == 0:
                id_placa = []
                id_placa.append(id)
                id_placa.append(idFrame[id])
                id_placa.append(arrayPlaca)
                self.listaPlacas.append(id_placa)
            # Se houver, precisa verificar se a placa ja está sendo rastreada
            # Se sim, adiciona ao seu ID em listaPlaca
            else:
                for i in range(len(self.listaPlacas)):
                    if self.listaPlacas[i][0] == id:
                        arrayPlaca = self.listaPlacas[i][1]
                        # Cada vez que o caracter se repete naquela posição 
                        # incrementa 1 na matriz
                        for j in range(7):
                            ind = ord(placa[j]) - ord("0")
                            arrayPlaca[j][ind] += 1
                        self.listaPlacas[i][1] = arrayPlaca
                        encontrou = True
                # Se após percorrer toda listaPlaca não achar é um ID novo,
                # então adiciona ID e placa em listaPlaca
                if not encontrou:
                    id_placa = []
                    id_placa.append(id)
                    id_placa.append(idFrame[id])
                    id_placa.append(arrayPlaca)
                    self.listaPlacas.append(id_placa)
        # Verifica se existe uma placa que deixou de ser rastreada
        ind = None
        for i in range(len(self.listaPlacas)):
            rastreando = False
            for (objectID, centroidTracking) in objectsTracking.items():
                if self.listaPlacas[i][0] == objectID:
                    rastreando = True
            if not rastreando:
                # Guarda a matriz que contem a placa e a frequencia
                # com que os caracteres aparecem
                placaFinal.append(self.listaPlacas[i])             
                ind = i
        if ind != None:
            del(self.listaPlacas[ind])
        return placaFinal

    def _run_local_storage_worker(self, index, storage_queue, plate_callback, error_callback):
        PASSWORD_CHARS = string.ascii_letters + string.digits
        PASSWORD_SIZE = 8
        self.logger.debug('%s\tlocal storage worker #%d - begin' % (self.tag_slug, index))
        try:
            data = storage_queue.get()
            while data is not None:
                if plate_callback is not None: plate_callback(self, data)
                try:
                    success, png = cv2.imencode('.png', data['roi'])
                    if not success: raise Exception('Error converting to PNG')
                    info = data['info']
                    filename = str(uuid.uuid4()) + '.zip'
                    password = ''.join(random.choice(PASSWORD_CHARS) for _ in range(PASSWORD_SIZE))
                    with zipfile.ZipFile(path.join(self._local_storage_dir, filename), mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('info.json', json.dumps(info))
                        zf.writestr('roi.png', png.tostring())
                    with open(path.join(self._local_storage_dir, filename), mode='rb') as fp:
                        md5sum = hashlib.md5()
                        for buf in iter(partial(fp.read, 128), b''):
                            md5sum.update(buf)
                    with self._local_storage_lock:
                        with open(path.join(self._local_storage_dir, filename + '.json'), 'w') as fp:
                            json.dump({
                                'who': self.tag_slug,
                                'when': info['when'],
                                'plate': info['plate'],
                                'filename': filename,
                                'password': password,
                                'md5sum': md5sum.hexdigest(),
                            }, fp)
                except Exception as error:
                    self.logger.error('%s\tlocal storage worker #%d - exception handler, loop still running: "%s"' % (self.tag_slug, index, str(error)))
                    if error_callback is not None: error_callback(self, {'error': error})
                finally:
                    storage_queue.task_done()
                data = storage_queue.get()
            storage_queue.task_done()
        except:
            self.logger.exception('%s\tlocal storage worker #%d - end by exception' % (self.tag_slug, index))
            raise            
        self.logger.debug('%s\tlocal storage worker #%d - end' % (self.tag_slug, index))

    def _run_final_storage_worker(self, error_callback):
        self.logger.debug('%s\tfinal storage worker - begin' % (self.tag_slug, ))
        try:
            while self._keep_running:
                try:
                    while self._keep_running:
                        files = glob(path.join(self._local_storage_dir, '*.json'))
                        if len(files) == 0:
                            time.sleep(1)
                            continue
                        with self._local_storage_lock:
                            json_filename = path.basename(files[0])
                        with open(path.join(self._local_storage_dir, json_filename), 'r') as fp:
                            data = json.load(fp)
                            self.send({
                                'type': 'agent-report',
                                'event': 'detection',
                                'who': self.tag_slug,
                                'data': data,
                            })
                            rename(path.join(self._local_storage_dir, data['filename']), path.join(self._final_storage_dir, data['filename']))
                            remove(path.join(self._local_storage_dir, json_filename))
                except Exception as error:
                    self.logger.error('%s\tfinal storage worker - exception handler, loop still running: "%s"' % (self.tag_slug, str(error)))
                    if error_callback is not None: error_callback(self, {'error': error})
                    time.sleep(10)
        except:
            self.logger.exception('%s\tfinal storage worker - end by exception' % (self.tag_slug, ))
            raise            
        self.logger.debug('%s\tfinal storage worker - end' % (self.tag_slug, ))

    def _run_processing(self, video_uri, num_frame_workers, num_local_storage_workers, frame_queue_size, storage_queue_size):
        self.logger.debug('%s\tprocessing thread - begin' % self.tag_slug)
        try:
            open_callback = processing.CALLBACKS.get('open')
            frame_callback = processing.CALLBACKS.get('frame')
            plate_callback = processing.CALLBACKS.get('plate')
            close_callback = processing.CALLBACKS.get('close')
            error_callback = processing.CALLBACKS.get('error')
            
            frame_queue = Queue(maxsize=frame_queue_size)
            storage_queue = Queue(maxsize=storage_queue_size)
    
            frame_workers = [threading.Thread(target=self._run_frame_worker, args=(index, frame_queue, storage_queue, frame_callback, ), daemon=True) for index in range(num_frame_workers)]
            for worker in frame_workers: worker.start()
                
            local_storage_workers = [threading.Thread(target=self._run_local_storage_worker, args=(index, storage_queue, plate_callback, error_callback, ), daemon=True) for index in range(num_local_storage_workers)]
            for worker in local_storage_workers: worker.start()

            final_storage_worker = threading.Thread(target=self._run_final_storage_worker, args=(error_callback, ), daemon=True)
            final_storage_worker.start()
                
            while self._keep_running:
                try:
                    self.logger.info('%s\tstream - connecting...' % self.tag_slug)
                    self._stream = cv2.VideoCapture(video_uri)
                    if not self._stream.isOpened(): raise Exception('Error opening camera')
                    try:
                        self.logger.info('%s\tstream - opened' % self.tag_slug)
                        if open_callback is not None: open_callback(self, {})
                        while self._keep_running and self._stream.isOpened():
                            grabbed, frame = self._stream.read()
                            if not grabbed: break
                            when = datetime.now().isoformat(sep=' ')
                            frame_queue.put((frame, when))
                        self.logger.info('%s\tstream - closed' % self.tag_slug)
                    finally:
                        if close_callback is not None: close_callback(self, {})
                        self._stream.release()
                except Exception as error:
                    self.logger.error('%s\tstream - exception handler, loop still running: "%s"' % (self.tag_slug, str(error)))
                    if error_callback is not None: error_callback(self, {'error': error})
                    time.sleep(10)
    
            for _ in range(num_frame_workers): frame_queue.put((None, None))
            frame_queue.join()
            
            for _ in range(num_local_storage_workers): storage_queue.put(None)
            storage_queue.join()
        except:
            self.logger.exception('%s\tprocessing thread - end by exception' % self.tag_slug)
            raise
        self.logger.debug('%s\tprocessing thread - end' % self.tag_slug)

    def _run_processing_rate(self):
        while self._keep_running:
            with self._processing_rate_lock:
                self._processing_counter = 0
                self._processing_counter_start = time.time()
            time.sleep(10)
            with self._processing_rate_lock:
                self._processing_rate = self._processing_counter / (time.time() - self._processing_counter_start) 
        with self._processing_rate_lock:
            self._processing_counter = 0
            self._processing_counter_start = None
            self._processing_rate = 0.

    def status(self):
        result = Agent.STATUS_STOPPED
        if self._stream is not None and self._stream.isOpened():
            result |= Agent.STATUS_CAMERA_RUNNING
        return result

    def processing_rate(self):
        with self._processing_rate_lock:
            result = self._processing_rate
        return result

    def run(self, server_address, video_uri, num_frame_workers=2, num_local_storage_workers=2, frame_queue_size=120, storage_queue_size=120):
        self.logger.info('%s\trun - begin [video_uri="%s", num_frame_workers=%d, num_local_storage_workers=%d, frame_queue_size=%d, storage_queue_size=%d]' % (self.tag_slug, video_uri, num_frame_workers, num_local_storage_workers, frame_queue_size, storage_queue_size))
        self._keep_running = True
        try:
            if self.status() != Agent.STATUS_STOPPED: raise Exception('The agent is running already')
            running_threads = [
                threading.Thread(target=self._run_processing, args=(video_uri, num_frame_workers, num_local_storage_workers, frame_queue_size, storage_queue_size, ), daemon=True),
                threading.Thread(target=self._run_processing_rate, daemon=True),
            ]
            for thread in running_threads: thread.start()
            for thread in running_threads: thread.join()
        except:
            self.logger.exception('%s\trun - end by exception' % self.tag_slug)
            raise
        finally:
            self._keep_running = False
        self.logger.info('%s\trun - end' % self.tag_slug)

    def stop(self):
        self._keep_running = False

    def send(self, data):
        self.logger.debug('%s\tsend (%s) - begin' % (self.tag_slug, data['type']))
        try:
            print(f"send -> {data}")  #TODO Ticket
        except:
            self.logger.exception('%s\tsend - end by exception' % self.tag_slug)
            raise
        self.logger.debug('%s\tsend (%s) - end' % (self.tag_slug, data['type']))
