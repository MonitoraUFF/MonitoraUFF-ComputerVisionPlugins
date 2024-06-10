import matplotlib
matplotlib.use('Agg', force=True)

from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum, IntFlag, unique
from easyocr import Reader
from functools import partial
from glob import glob
from numpy.typing import NDArray
from queue import Queue
from scipy.spatial import distance
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict
from ultralytics import YOLO
import cv2
import hashlib, json, logging, logging.config, os, random, string, threading, time, uuid, zipfile
import numpy as np
import torch


DEFAULT_PATIENCE_IN_SECONDS: Final = 2.0
DEFAULT_NUM_VEHICLE_WORKERS: Final = 4
DEFAULT_NUM_LOCAL_STORAGE_WORKERS: Final = 2
DEFAULT_VEHICLE_QUEUE_SIZE: Final = 300
DEFAULT_STORAGE_QUEUE_SIZE: Final = 300


class DetectedVehicle(TypedDict):
    vehicle: NDArray[np.uint8]
    vehicle_roi: Tuple[int, int, int, int]
    when: datetime


class DetectedPlate(TypedDict):
    plate: str
    warped_plate: NDArray[np.uint8]
    vehicle: NDArray[np.uint8]
    vehicle_roi: Tuple[int, int, int, int]
    plate_roi: Tuple[int, int, int, int]
    when: datetime


class ReportedPlate(TypedDict):
    plate: str
    vehicle: NDArray[np.uint8]
    warped_plate: NDArray[np.uint8]
    when: datetime


class StoredPlate(TypedDict):
    plate: str
    when: datetime


class TrackedVehicle(TypedDict):
    centroid: NDArray[np.float32]
    history: List[DetectedPlate]


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

    CHAR_WHITELIST: Final = ''.join(map(chr, range(ord('0'), ord('9')+1))) + \
                            ''.join(map(chr, range(ord('A'), ord('Z')+1)))

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

    def _find_plate_candidates(self, lpd_model: YOLO, ocr_model: Reader, detected_vehicle: DetectedVehicle) -> List[DetectedPlate]:
        result: List[DetectedPlate] = list()
        # The vehicle's image may include several plate box candidates
        vehicle_bgr = detected_vehicle['vehicle']
        for plate_box in lpd_model(vehicle_bgr, device=lpd_model.device, verbose=True)[0].boxes:
            # The plate box includes a perspective projection of the plate, so we have to find the quadrilateral resulting from this projection
            x1, y1, x2, y2 = map(lambda arg: arg.item(), plate_box.xyxy.round().to(torch.int32)[0].cpu())
            plate_bgr = vehicle_bgr[y1:y2, x1:x2]
            plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
            plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(plate_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            # Each contour is a candidate projected plate
            for contour in contours:
                # But the contours may be defined by por than four vertices, so we have to simplify it
                chull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
                while len(chull) > 4:
                    vertices = np.concatenate((np.concatenate((chull[-1][None, ...], chull, chull[0][None, ...]), axis=0).squeeze(), np.ones((len(chull) + 2, 1), dtype=chull.dtype)), axis=1)
                    matrices = np.stack((vertices[:-2], vertices[1:-1], vertices[2:]), axis=1, dtype=np.float32)
                    min_idx = np.argmin(np.abs(np.linalg.det(matrices)))
                    chull = np.delete(chull, min_idx, axis=0)
                # Given a convex quadrilateral...
                if len(chull) == 4:
                    # ... warp it to a rectangle...
                    x = chull.squeeze()
                    mu = x.mean(axis=0)
                    sigma = np.cov(x.T)
                    u, _, _ = np.linalg.svd(sigma)
                    if u[0, 0] < 0: u[:, 0] *= -1
                    if u[1, 1] < 0: u[:, 1] *= -1
                    d = np.matmul(x - mu, u)
                    angle = np.arctan2(d[:, 1], d[:, 0])
                    source_pts = np.float32(x[np.argsort(angle)])
                    target_pts = np.float32([[0, 0], [200, 0], [200, 65], [0, 65]])
                    affine_mtx = cv2.estimateAffine2D(source_pts, target_pts)[0]
                    warped_plate_bgr = cv2.warpAffine(plate_bgr, affine_mtx, (200, 65))
                    warped_plate_gray = cv2.warpAffine(plate_gray, affine_mtx, (200, 65))
                    # ... find each character...
                    #TODO Substituindo o OCR feito sobre a imagem crua, pois ele não funciona
                    # ... and perform OCR
                    plate = ''.join(ocr_model.readtext(warped_plate_gray, detail=0, allowlist=self.CHAR_WHITELIST))
                    if len(plate) > 0:
                        result.append(DetectedPlate(
                            plate=plate,
                            warped_plate=warped_plate_bgr,
                            vehicle=vehicle_bgr,
                            vehicle_roi=detected_vehicle['vehicle_roi'],
                            plate_roi=(x1, y1, x2, y2),
                            when=detected_vehicle['when'],
                        ))
        return result
    
    def _make_reported_plate(self, history: List[DetectedPlate]) -> ReportedPlate:
        # Check the frequency of each character in each entry of the plate 
        num_entries = max(map(lambda arg: len(arg['plate']), history))
        count = np.zeros((num_entries, len(self.CHAR_WHITELIST)), dtype=np.uint32)
        for detected_plate in history:
            for entry_idx, entry in enumerate(detected_plate['plate']):
                count[entry_idx, self.CHAR_WHITELIST.index(entry)] += 1
        # Keep the most voted characters
        winner_idx = count.argmax(axis=1)
        plate = ''.join(map(lambda idx: self.CHAR_WHITELIST[idx], winner_idx))
        # Find the most likely image
        sequence_matcher = SequenceMatcher(a=plate)
        best_ratio = -float("inf")
        best_detected_plate = None
        for detected_plate in history:
            sequence_matcher.set_seq2(detected_plate['plate'])
            ratio = sequence_matcher.ratio()
            if best_ratio < ratio:
                best_ratio = ratio
                best_detected_plate = detected_plate
        # Return the plate to be reported
        return ReportedPlate(
            plate=plate,
            vehicle=best_detected_plate['vehicle'],
            warped_plate=best_detected_plate['warped_plate'],
            when=best_detected_plate['when'],
        )
    
    def _perform_tracking(self, detected_plates: List[DetectedPlate], patience_in_seconds: float) -> List[ReportedPlate]:
        result: List[ReportedPlate] = list()
        with self._tracking_lock:
            # If there are vehicles being tracked then they must be updated or deregistred
            num_tracked_vehicles = len(self._tracked_vehicles)
            if num_tracked_vehicles != 0:
                # In case of new plates, the existing vehicles may be updated or deregistred
                num_detected_plates = len(detected_plates)
                if num_detected_plates != 0:
                    # Create the list of centroids of the plates detected in the current frame
                    new_centroids = np.empty((num_detected_plates, 2), dtype=np.float32)
                    for idx, detected_plate in enumerate(detected_plates):
                        offset_x, offset_y, _, _ = detected_plate['vehicle_roi']
                        x1, y1, x2, y2 = detected_plate['plate_roi']
                        new_centroids[idx] = (offset_x + (x1 + x2) / 2, offset_y + (y1 + y2) / 2)
                    # Create the list of centroids of the plates of tracked vehicles
                    old_centroids = np.empty((num_tracked_vehicles, 2), dtype=np.float32)
                    for idx, tracked_vehicle in enumerate(self._tracked_vehicles):
                        old_centroids[idx] = tracked_vehicle['centroid']
                    # Compute Euclidean distance between each pair of the two collections ofcentroids
                    dist = distance.cdist(old_centroids, new_centroids)
                    # To check correspondence, we have to find the smallest value in each row and
                    # sort the row indices based on their minimum values ​​so that the row with the
                    # smallest value is in front of the index
                    rows = dist.min(axis=1).argsort()
                    # Next, we perform a similar process on the columns by finding the smallest value
                    # in each column and then sorting using the previously computed row index list
                    cols = dist.argmin(axis=1)[rows]
                    # To determine whether we need to update, register, or deregister a vehicle we
                    # need to keep track of which of the row and column indexes we have already examined
                    used_rows = np.zeros((num_tracked_vehicles,), dtype=np.bool_)
                    used_cols = np.zeros((num_detected_plates,), dtype=np.bool_)
                    # Iterate over the combinations of rows and columns to update the tracked vehicle history
                    for row, col in zip(rows, cols):
                        # If we have already looked at the row or column value before, ignore it
                        if used_rows[row] or used_cols[col]:
                            continue
                        # Otherwise, we must update the tracked vehicle history
                        tracked_vehicle = self._tracked_vehicles[row]
                        tracked_vehicle['centroid'] = new_centroids[col]
                        tracked_vehicle['history'].append(detected_plates[col])
                        used_rows[row] = True
                        used_cols[col] = True
                    # Iterate over the unused rows and deregister vehicles that have been
                    # missing for a long time
                    for row in range(num_tracked_vehicles-1, -1, -1):
                        if not used_rows[row]:
                            tracked_vehicle = self._tracked_vehicles[row]
                            elapsed_time = datetime.now() - tracked_vehicle['history'][-1]['when']
                            if elapsed_time.total_seconds() > patience_in_seconds:
                                result.append(self._make_reported_plate(tracked_vehicle['history']))
                                del self._tracked_vehicles[row]
                    # Iterate over the unused columns and register new vehicles
                    for col in range(num_detected_plates):
                        if not used_cols[col]:
                            self._tracked_vehicles.append(TrackedVehicle(
                                centroid=new_centroids[col],
                                history=[detected_plates[col]],
                            ))
                # Otherwise, existing vehicles may only be deregristred if they have been missing for a long time
                else:
                    for idx in range(num_tracked_vehicles-1, -1, -1):
                        tracked_vehicle = self._tracked_vehicles[idx]
                        elapsed_time = datetime.now() - tracked_vehicle['history'][-1]['when']
                        if elapsed_time.total_seconds() > patience_in_seconds:
                            result.append(self._make_reported_plate(tracked_vehicle['history']))
                            del self._tracked_vehicles[idx]
            # Otherwise, the detected plates are new vehicles to be tracked 
            else:
                for detected_plate in detected_plates:
                    offset_x, offset_y, _, _ = detected_plate['vehicle_roi']
                    x1, y1, x2, y2 = detected_plate['plate_roi']
                    centroid = np.asarray((offset_x + (x1 + x2) / 2, offset_y + (y1 + y2) / 2), dtype=np.float32)
                    self._tracked_vehicles.append(TrackedVehicle(
                        centroid=centroid,
                        history=[detected_plate],
                    ))
        # Report the missing vehicles
        return result

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

    def _run_processing(self, server_address: str, video_uri: str, patience_in_seconds: float, num_vehicle_workers: int, num_local_storage_workers: int, vehicle_queue_size: int, storage_queue_size: int) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        self._status |= AgentStatus.RUNNING
        self._logger.debug(f'{self.tag_slug}\tprocessing thread - begin (using "{device}")')
        try:
            # Create the queues that keep the data to be processed and stored locally
            vehicle_queue: "Queue[DetectedVehicle]" = Queue(maxsize=vehicle_queue_size)
            storage_queue: "Queue[ReportedPlate]" = Queue(maxsize=storage_queue_size)
            # Create a set of YOLO models pre-trained on a LPR dataset (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and a set of EasyOCR models
            lpd_models = [YOLO(os.path.join(config_dir, 'yolov8lpd.pt')).to(device) for _ in range(num_vehicle_workers)]
            ocr_models = [Reader(['en'], gpu=torch.cuda.is_available()) for _ in range(num_vehicle_workers)]
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
                        # For each frame...
                        for vehicles in coco_model(video_uri, classes=used_coco_ids, device=coco_model.device, stream=True, verbose=True):
                            if not self._keep_running:
                                break
                            begin = time.perf_counter()
                            frame_bgr = vehicles.orig_img
                            # ... get each detected vehicle and enqueue them for processing
                            for vehicle_box in vehicles.boxes:
                                x1, y1, x2, y2 = map(lambda arg: arg.item(), vehicle_box.xyxy.round().to(torch.int32)[0])
                                vehicle_bgr = frame_bgr[y1:y2, x1:x2]
                                vehicle_queue.put(DetectedVehicle(
                                    vehicle=vehicle_bgr,
                                    vehicle_roi=(x1, y1, x2, y2),
                                    when=datetime.now())
                                )
                            # Enqueue empty data to sinalize the end of processing
                            for _ in range(num_vehicle_workers): vehicle_queue.put(None)
                            # Create and start vehicle processing workers using YOLO models pre-trained on a LPR dataset (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
                            detected_plates: List[DetectedPlate] = list()
                            vehicle_workers = [threading.Thread(target=self._run_vehicle_worker, args=(worker_idx, lpd_models[worker_idx], ocr_models[worker_idx], vehicle_queue, detected_plates,), daemon=True) for worker_idx in range(num_vehicle_workers)]
                            elapsed_initializetion = time.perf_counter() - begin
                            begin = time.perf_counter()
                            for worker in vehicle_workers: worker.start()
                            # Wait for the end of vehicles' processing and update the tracking system with the plate candidate and retrieve the plates to be reported
                            vehicle_queue.join()
                            elapsed_processing = time.perf_counter() - begin
                            begin = time.perf_counter()
                            for reported_plate in self._perform_tracking(detected_plates, patience_in_seconds):
                                storage_queue.put(reported_plate)
                            elapsed_tracking = time.perf_counter() - begin
                            print(f'Initialization: {elapsed_initializetion:1.4f}, Processing: {elapsed_processing:1.4f}, Tracking: {elapsed_tracking:1.4f}, Total: {elapsed_initializetion + elapsed_processing + elapsed_tracking:1.4f}')
                        self._logger.info(f'{self.tag_slug}\tstream closed')
                    finally:
                        self._send(server_address, AgentEvent.STREAM_CLOSE, None)
                        self._status ^= AgentStatus.PROCESSING
                except Exception as error:
                    self._logger.error(f'{self.tag_slug}\tstream closed by exception, loop still running: "{str(error)}"')
                    time.sleep(10)
            for _ in range(num_local_storage_workers): storage_queue.put(None)
            storage_queue.join()
            self._logger.debug(f'{self.tag_slug}\tprocessing thread - end')
        except Exception as error:
            self._logger.exception(f'{self.tag_slug}\tprocessing thread - end by exception: "{str(error)}')
            raise
        finally:
            self._status ^= AgentStatus.RUNNING

    def _run_vehicle_worker(self, worker_idx: int, lpd_model: YOLO, ocr_model: Reader, vehicle_queue: "Queue[DetectedVehicle]", detected_plates: List[DetectedPlate]) -> None:
        # Get first vehicle
        detected_vehicle = vehicle_queue.get()
        try:
            # If there is a vehicle to be processed then...
            while detected_vehicle is not None:
                vehicle_queue.task_done()
                # .... find each plate candidate in the vehicle's image...
                plate_candidates = self._find_plate_candidates(lpd_model, ocr_model, detected_vehicle)
                # ... and update the list of detected plates if needed
                if len(plate_candidates) != 0:
                    plate_candidates.sort(key=lambda arg: len(arg['plate']))
                    detected_plates.append(plate_candidates[-1])
                # Get the next vehicle
                detected_vehicle = vehicle_queue.get()
        except Exception as error:
            self._logger.exception(f'{self.tag_slug}\vehicle processing worker #{worker_idx} - end by exception: "{str(error)}"')
        finally:
            vehicle_queue.task_done()
        
    def _send(self, server_address: str, event: AgentEvent, data: Optional[Dict[str, Any]]) -> None:
        self._logger.debug(f'{self.tag_slug}\tsend - begin')
        try:
            print(f'licenceplate agent "{self.tag_slug}" reported the event "{event}" and sent the following data to "{server_address}" -> {data}')  #TODO Ticket
        except:
            self._logger.exception(f'{self.tag_slug}\tsend - end by exception')
            raise
        self._logger.debug(f'{self.tag_slug}\tsend - end')

    def run(self, server_address: str, video_uri: str, patience_in_seconds: float = DEFAULT_PATIENCE_IN_SECONDS, num_vehicle_workers: int = DEFAULT_NUM_VEHICLE_WORKERS, num_local_storage_workers: int = DEFAULT_NUM_LOCAL_STORAGE_WORKERS, vehicle_queue_size: int = DEFAULT_VEHICLE_QUEUE_SIZE, storage_queue_size: int = DEFAULT_STORAGE_QUEUE_SIZE) -> None:
        self._logger.info(f'{self.tag_slug}\trun - begin [video_uri="{video_uri}", num_vehicle_workers={num_vehicle_workers}, num_local_storage_workers={num_local_storage_workers}, vehicle_queue_size={vehicle_queue_size}, storage_queue_size={storage_queue_size}]')
        try:
            # Check if the agent is already running
            if AgentStatus.RUNNING in self.status: raise Exception('The agent is running already')
            self._keep_running = True
            try:
                # Start the main thread of the agent
                thread = threading.Thread(target=self._run_processing, args=(server_address, video_uri, patience_in_seconds, num_vehicle_workers, num_local_storage_workers, vehicle_queue_size, storage_queue_size,), daemon=True)
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
