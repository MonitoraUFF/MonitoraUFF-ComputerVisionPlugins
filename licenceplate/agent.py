import matplotlib
matplotlib.use('Agg', force=True)

from datetime import datetime
from enum import Enum, IntFlag, unique
from functools import partial
from glob import glob
from numpy.typing import NDArray
from queue import Queue
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict
from ultralytics import YOLO
import cv2
import hashlib, json, logging, logging.config, os, random, string, threading, time, uuid, zipfile
import numpy as np
import pytesseract
import torch


DEFAULT_NUM_VEHICLE_WORKERS: Final = 8
DEFAULT_NUM_LOCAL_STORAGE_WORKERS: Final = 2
DEFAULT_VEHICLE_QUEUE_SIZE: Final = 300
DEFAULT_STORAGE_QUEUE_SIZE: Final = 300


class DetectedVehicle(TypedDict):
    vehicle: NDArray[np.uint8]
    when: datetime


class DetectedPlate(TypedDict):  #TODO Verificar tipos
    plate: str
    warped_plate: NDArray[np.uint8]
    vehicle: NDArray[np.uint8]
    roi: Tuple[int, int, int, int]
    when: datetime


class ReportedPlate(TypedDict):
    plate: str
    warped_plate: NDArray[np.uint8]
    when: datetime


class StoredPlate(TypedDict):
    plate: str
    when: datetime


class TrackedVehicle(TypedDict):
    pass


@unique
class AgentEvent(Enum):
    DETECTION = 0
    STREAM_OPEN = 1
    STREAM_CLOSE = 2


class AgentStatus(IntFlag):
    STOPPED = 0
    RUNNING = 1
    PROCESSING = 2


class Agent:

    def __init__(self, tag_slug: str) -> None:
        # Set the logging system
        logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'logging.conf'))
        self._logger = logging.getLogger('agent')
        # Initialize basic attributes
        self.tag_slug: Final = tag_slug
        self._keep_running = False
        self._status = AgentStatus.STOPPED
        # Initialize tracking attributes
        self._tracking_lock = threading.Lock()
        self._tracked_vehicles: List[TrackedVehicle] = list()
        # Initialize storage attributes
        self._local_storage_lock = threading.Lock()
        self._local_storage_dir = os.path.join('plugins-storage', 'licenceplate', 'local', tag_slug)
        os.makedirs(self._local_storage_dir, exist_ok=True)
        self._final_storage_dir = os.path.join('plugins-storage', 'licenceplate', 'final', tag_slug)
        os.makedirs(self._final_storage_dir, exist_ok=True)
        # Log agent initialization
        self._logger.info(f'{self.tag_slug}\tinit')

    def __del__(self) -> None:
        # Log agent finalization
        self._logger.info(f'{self.tag_slug}\tdel')

    def _perform_tracking(self, detected_plate: DetectedPlate) -> List[ReportedPlate]:
        return [ReportedPlate(
                    plate=detected_plate['plate'],
                    vehicle=detected_plate['vehicle'],
                    warped_plate=detected_plate['warped_plate'],
                    when=detected_plate['when'],
                )]

    def _run_final_storage_worker(self, server_address: str) -> None:
        self._logger.debug(f'{self.tag_slug}\tfinal storage worker - begin')
        try:
            # Set a loop to restart the final storage process in case of some exception
            while self._keep_running:
                try:
                    while self._keep_running:
                        # When a new file is found...
                        files = glob(os.path.join(self._local_storage_dir, '*.json'))
                        if len(files) == 0:
                            time.sleep(1)
                            continue
                        # ... copy the file from the local to the final storage
                        with self._local_storage_lock:
                            json_filename = os.path.basename(files[0])
                            with open(os.path.join(self._local_storage_dir, json_filename), 'r') as fp:
                                data = json.load(fp)
                            self._send(server_address, AgentEvent.DETECTION, data)
                            os.rename(os.path.join(self._local_storage_dir, data['filename']), os.path.join(self._final_storage_dir, data['filename']))
                            os.remove(os.path.join(self._local_storage_dir, json_filename))
                except Exception as error:
                    self._logger.error(f'{self.tag_slug}\tfinal storage worker - exception handler, loop still running: "{str(error)}"')
                    time.sleep(10)
        except:
            self._logger.exception(f'{self.tag_slug}\tfinal storage worker - end by exception')
            raise            
        self._logger.debug(f'{self.tag_slug}\tfinal storage worker - end')

    def _run_local_storage_worker(self, worker_idx: int, storage_queue: "Queue[ReportedPlate]") -> None:
        PASSWORD_CHARS = string.ascii_letters + string.digits
        PASSWORD_SIZE = 8
        self._logger.debug(f'{self.tag_slug}\tlocal storage worker #{worker_idx} - begin')
        try:
            # While we have data to store...
            reported_plate = storage_queue.get()
            while reported_plate is not None:
                try:
                    # Encode the plate as a PNG file
                    success, vehicle_png = cv2.imencode('.png', reported_plate['vehicle'])
                    if not success: raise Exception('Error converting to PNG')
                    success, warped_plate_png = cv2.imencode('.png', reported_plate['warped_plate'])
                    if not success: raise Exception('Error converting to PNG')
                    # Put all information in a ZIP file
                    filename = f'{str(uuid.uuid4())}.zip'
                    password = ''.join(random.choice(PASSWORD_CHARS) for _ in range(PASSWORD_SIZE))
                    info = StoredPlate(
                        plate=reported_plate['plate'],
                        when=reported_plate['when'].isoformat(sep=' '),
                    )
                    with zipfile.ZipFile(os.path.join(self._local_storage_dir, filename), mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('info.json', json.dumps(info))
                        zf.writestr('vehicle.png', vehicle_png.tostring())
                        zf.writestr('warped_plate.png', warped_plate_png.tostring())
                    with open(os.path.join(self._local_storage_dir, filename), mode='rb') as fp:
                        md5sum = hashlib.md5()
                        for buf in iter(partial(fp.read, 128), b''):
                            md5sum.update(buf)
                    # Write a header to a JSON file
                    with self._local_storage_lock:
                        with open(os.path.join(self._local_storage_dir, f'{filename}.json'), 'w') as fp:
                            json.dump({
                                'who': self.tag_slug,
                                'when': info['when'],
                                'plate': info['plate'],
                                'filename': filename,
                                'password': password,
                                'md5sum': md5sum.hexdigest(),
                            }, fp)
                except Exception as error:
                    self._logger.error(f'{self.tag_slug}\tlocal storage worker #{worker_idx} - exception handler, loop still running: "{str(error)}"')
                finally:
                    storage_queue.task_done()
                reported_plate = storage_queue.get()
            storage_queue.task_done()
        except:
            self._logger.exception(f'{self.tag_slug}\tlocal storage worker #{worker_idx} - end by exception')
            raise            
        self._logger.debug(f'{self.tag_slug}\tlocal storage worker #{worker_idx} - end')

    def _run_processing(self, server_address: str, video_uri: str, num_vehicle_workers: int, num_local_storage_workers: int, vehicle_queue_size: int, storage_queue_size: int) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        self._status |= AgentStatus.RUNNING
        self._logger.debug(f'{self.tag_slug}\tprocessing thread - begin')
        try:
            # Create the queues that keep the data to be processed and stored locally
            vehicle_queue: "Queue[DetectedVehicle]" = Queue(maxsize=vehicle_queue_size)
            storage_queue: "Queue[ReportedPlate]" = Queue(maxsize=storage_queue_size)
            # Create and start vehicle processing workers using YOLO models pre-trained on a LPR dataset (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
            vehicle_workers = [threading.Thread(target=self._run_vehicle_worker, args=(worker_idx, YOLO(os.path.join(config_dir, 'yolov8lpd.pt')).to(device), vehicle_queue, storage_queue,), daemon=True) for worker_idx in range(num_vehicle_workers)]
            for worker in vehicle_workers: worker.start()
            # Create and start local storage workers
            local_storage_workers = [threading.Thread(target=self._run_local_storage_worker, args=(worker_idx, storage_queue,), daemon=True) for worker_idx in range(num_local_storage_workers)]
            for worker in local_storage_workers: worker.start()
            # Create and start the final storage worker
            final_storage_worker = threading.Thread(target=self._run_final_storage_worker, args=(server_address,), daemon=True)
            final_storage_worker.start()
            # Load a YOLO model pre-trained on the COCO dataset and find the ID of the vehicle classes
            coco_model = YOLO(os.path.join(config_dir, 'yolov8n.pt')).to(device)
            coco_ids = list(coco_model.names.keys())
            coco_names = list(coco_model.names.values())
            used_coco_ids = [coco_ids[coco_names.index(name)] for name in ['car', 'motorcycle', 'bus', 'truck']]
            # Set a loop to restart the video processing in case of some exception
            while self._keep_running:
                try:
                    # Open the video stream
                    self._status |= AgentStatus.PROCESSING 
                    self._send(server_address, AgentEvent.STREAM_OPEN, None)
                    self._logger.info(f'{self.tag_slug}\tstream opened')
                    try:
                        for vehicles in coco_model(video_uri, classes=used_coco_ids, device=coco_model.device, stream=True, verbose=True):
                            if not self._keep_running:
                                break
                            frame_bgr = vehicles.orig_img
                            # ... get each detected vehicle...
                            for vehicle_box in vehicles.boxes:
                                x1, y1, x2, y2 = vehicle_box.xyxy.round().to(torch.int32)[0]
                                vehicle_bgr = frame_bgr[y1:y2, x1:x2, :]
                                vehicle_queue.put(DetectedVehicle(vehicle=vehicle_bgr, when=datetime.now()))
                        self._logger.info(f'{self.tag_slug}\tstream closed')
                    finally:
                        self._send(server_address, AgentEvent.STREAM_CLOSE, None)
                        self._status ^= AgentStatus.PROCESSING
                except Exception as error:
                    self._logger.error(f'{self.tag_slug}\tstream closed by exception, loop still running: "{str(error)}"')
                    time.sleep(10)
            # Enqueue empty data to sinalize the end of processing
            for _ in range(num_vehicle_workers): vehicle_queue.put(None)
            vehicle_queue.join()
            for _ in range(num_local_storage_workers): storage_queue.put(None)
            storage_queue.join()
            self._logger.debug(f'{self.tag_slug}\tprocessing thread - end')
        except Exception as error:
            self._logger.exception(f'{self.tag_slug}\tprocessing thread - end by exception: "{str(error)}')
            raise
        finally:
            self._status ^= AgentStatus.RUNNING

    def _run_vehicle_worker(self, worker_idx: int, lpd_model: YOLO, vehicle_queue: "Queue[DetectedVehicle]", storage_queue: "Queue[ReportedPlate]") -> None:
        CHAR_WHITELIST: Final = ''.join(map(chr, range(ord('0'), ord('9')+1))) + \
                                ''.join(map(chr, range(ord('A'), ord('Z')+1))) + \
                                ''.join(map(chr, range(ord('a'), ord('z')+1)))
        self._logger.debug(f'{self.tag_slug}\vehicle procedding worker #{worker_idx} - begin')
        try:
            # Get vehicle
            detected_vehicle = vehicle_queue.get()
            while detected_vehicle is not None:
                vehicle_bgr, when = detected_vehicle['vehicle'], detected_vehicle['when']
                vehicle_queue.task_done()
                # Find each plate candidate in the vehicle
                plate_candidates: List[DetectedPlate] = list()
                for plate_box in lpd_model(vehicle_bgr, device=lpd_model.device, verbose=True)[0].boxes:
                    if not self._keep_running:
                        break
                    x1, y1, x2, y2 = map(lambda arg: arg.item(), plate_box.xyxy.round().to(torch.int32)[0].cpu())
                    plate_bgr = vehicle_bgr[y1:y2, x1:x2, :]
                    plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
                    plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    contours, _ = cv2.findContours(plate_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                    for contour in contours:
                        chull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
                        while len(chull) > 4:
                            vertices = np.concatenate((np.concatenate((chull[-1, ...][None, ...], chull, chull[0, ...][None, ...]), axis=0).squeeze(), np.ones((len(chull) + 2, 1), dtype=chull.dtype)), axis=1)
                            matrices = np.stack((vertices[:-2, :], vertices[1:-1, :], vertices[2:, :]), axis=1, dtype=np.float32)
                            min_idx = np.argmin(np.abs(np.linalg.det(matrices)))
                            chull = np.delete(chull, min_idx, axis=0)
                        # When we have a candidate plate...
                        if len(chull) == 4:
                            # ... warp its image...
                            x = chull.squeeze()
                            mu = x.mean(axis=0)
                            sigma = np.cov(x.T)
                            u, _, _ = np.linalg.svd(sigma)
                            if u[0, 0] < 0: u[:, 0] *= -1
                            if u[1, 1] < 0: u[:, 1] *= -1
                            d = np.matmul(x - mu, u)
                            angle = np.arctan2(d[:, 1], d[:, 0])
                            source_pts = np.float32(x[np.argsort(angle), :])
                            target_pts = np.float32([[0, 0], [200, 0], [200, 65], [0, 65]])
                            affine_mtx = cv2.estimateAffine2D(source_pts, target_pts)[0]
                            warped_plate_bgr = cv2.warpAffine(plate_bgr, affine_mtx, (200, 65))
                            # ... and perform OCR
                            plate = pytesseract.image_to_string(cv2.cvtColor(warped_plate_bgr, cv2.COLOR_BGR2RGB), config=f'--oem 3 --psm 8')
                            plate = ''.join(map(lambda arg: arg if arg in CHAR_WHITELIST else '', plate)).upper()
                            if len(plate) > 0:
                                plate_candidates.append(DetectedPlate(
                                    plate=plate,
                                    warped_plate=warped_plate_bgr,
                                    vehicle=vehicle_bgr,
                                    roi=(x1, y1, x2, y2),
                                    when=when,
                                ))
                # Update the tracking system and retrieve the plates to be reported
                if len(plate_candidates) != 0:
                    plate_candidates.sort(key=lambda arg: len(arg['plate']))
                    for reported_plate in self._perform_tracking(plate_candidates[-1]):
                        storage_queue.put(reported_plate)
                # Get the next vehicle
                detected_vehicle = vehicle_queue.get()
            vehicle_queue.task_done()
        except:
            self._logger.exception(f'{self.tag_slug}\vehicle processing worker #{worker_idx} - end by exception')
            raise            
        self._logger.debug(f'{self.tag_slug}\vehicle processing worker #{worker_idx} - end')
        
    def _send(self, server_address: str, event: AgentEvent, data: Optional[Dict[str, Any]]) -> None:
        self._logger.debug(f'{self.tag_slug}\tsend - begin')
        try:
            print(f'licenceplate agent "{self.tag_slug}" reported the event "{event}" and sent the following data to "{server_address}" -> {data}')  #TODO Ticket
        except:
            self._logger.exception(f'{self.tag_slug}\tsend - end by exception')
            raise
        self._logger.debug(f'{self.tag_slug}\tsend - end')

    def run(self, server_address: str, video_uri: str, num_vehicle_workers: int = DEFAULT_NUM_VEHICLE_WORKERS, num_local_storage_workers: int = DEFAULT_NUM_LOCAL_STORAGE_WORKERS, vehicle_queue_size: int = DEFAULT_VEHICLE_QUEUE_SIZE, storage_queue_size: int = DEFAULT_STORAGE_QUEUE_SIZE) -> None:
        self._logger.info(f'{self.tag_slug}\trun - begin [video_uri="{video_uri}", num_vehicle_workers={num_vehicle_workers}, num_local_storage_workers={num_local_storage_workers}, vehicle_queue_size={vehicle_queue_size}, storage_queue_size={storage_queue_size}]')
        try:
            # Check if the agent is already running
            if AgentStatus.RUNNING in self.status: raise Exception('The agent is running already')
            self._keep_running = True
            try:
                # Start the main thread of the agent
                thread = threading.Thread(target=self._run_processing, args=(server_address, video_uri, num_vehicle_workers, num_local_storage_workers, vehicle_queue_size, storage_queue_size,), daemon=True)
                thread.start()
                thread.join()
            finally:
                self._keep_running = False
        except:
            self._logger.exception(f'{self.tag_slug}\trun - end by exception')
            raise
        self._logger.info(f'{self.tag_slug}\trun - end')

    def stop(self) -> None:
        # Set the flag that terminates the loops
        self._keep_running = False

    @property
    def status(self) -> AgentStatus:
        return self._status
