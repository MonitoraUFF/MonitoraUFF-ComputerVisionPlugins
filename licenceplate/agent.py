import matplotlib
matplotlib.use('Agg', force=True)

from . import default
from cv2 import VideoCapture, VideoWriter
from datetime import datetime
from difflib import SequenceMatcher
from enum import auto, Enum, IntEnum, IntFlag, unique
from easyocr import Reader
from bytetracker import BYTETracker
from functools import partial
from glob import glob
from numpy.typing import NDArray
from queue import Queue
from tqdm import tqdm
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Tuple, TypedDict
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from zipfile import ZipFile
import cv2
import hashlib, json, logging, logging.config, os, random, string, threading, time, uuid, zipfile
import numpy as np
import torch


CHAR_WHITELIST: Final = ''.join(map(chr, range(ord('0'), ord('9')+1))) + ''.join(map(chr, range(ord('A'), ord('Z')+1)))


TARGET_HEIGHT: Final = 65
TARGET_WIDTH: Final = 200
TARGET_AREA: Final = TARGET_WIDTH * TARGET_HEIGHT
TARGET_CORNERS: Final = np.asarray([[0, 0], [TARGET_WIDTH, 0], [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]], dtype=np.float32)


USED_COCO_CLASS_NAMES: List[str] = ['car', 'bus', 'truck']  #TODO Implementar heurística para placa de 'motorcycle'


_hex_to_bgr = lambda hex: (int(hex[4:6], base=16), int(hex[2:4], base=16), int(hex[0:2], base=16))
DRAW_BGR_COLOR_PALETTE = list(map(_hex_to_bgr, ['9E0142', 'D53E4F', 'F46D43', 'FDAE61', 'FEE08B', 'E6F598', 'ABDDA4', '66C2A5', '3288BD', '5E4FA2',]))
DRAW_BGR_MEDIUM_BLUE = _hex_to_bgr('0077B6')
DRAW_BGR_PASTEL_PINK = _hex_to_bgr('FFADAD')
DRAW_BGR_PASTEL_PEACH = _hex_to_bgr('FFD6A5')
DRAW_BGR_PASTEL_YELLOW = _hex_to_bgr('FDFFB6')
DRAW_BGR_PASTEL_GREEN = _hex_to_bgr('CAFFBF')
DRAW_BGR_WHITE = _hex_to_bgr('FFFFFF')
DRAW_BOX_THICKNESS = 3
DRAW_TRACKING_THICKNESS = 2


@unique
class AgentEvent(Enum):
    DETECTION_CREATE = auto()
    DETECTION_UPDATE = auto()
    STREAM_OPEN = auto()
    STREAM_CLOSE = auto()


class AgentStatus(IntFlag):
    STOPPED = 0      # 00b -> The empty set
    RUNNING = 1      # 01b
    PROCESSING = 2   # 10b


ROI = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class DetectedPlate(TypedDict):
    plate: str
    warped_plate: NDArray[np.uint8]
    warped_plate_ocr: NDArray[np.uint8]
    vehicle: NDArray[np.uint8]
    vehicle_roi: ROI
    plate_roi: ROI
    when: datetime


class DetectedVehicle(TypedDict):
    vehicle: NDArray[np.uint8]
    vehicle_roi: ROI
    when: datetime


class ReportedPlate(TypedDict):
    id: str
    plate: str
    vehicle: NDArray[np.uint8]
    warped_plate: NDArray[np.uint8]
    warped_plate_ocr: NDArray[np.uint8]
    when: datetime


@unique
class StorageTask(IntEnum):
    CREATE = auto()
    UPDATE = auto()


class StoredPlate(TypedDict):
    plate: str
    when: datetime


class TrackedVehicle(TypedDict):
    id: str
    last_seen: datetime
    centroid_history: List[Tuple[int, int]]
    plate_history: List[DetectedPlate]


class Agent:

    def __init__(self, tag_slug: str) -> None:
        # Set the logging system
        logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'logging.conf'))
        self._logger = logging.getLogger('agent')
        # Initialize basic attributes
        self.tag_slug: Final = tag_slug
        self._keep_running = False
        self._status = AgentStatus.STOPPED
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

    def _draw_debug_info(self, frame_bgr: NDArray[np.uint8], vehicles: List[DetectedVehicle], plates: List[DetectedPlate], tracked_vehicles: Dict[int, TrackedVehicle]) -> NDArray[np.uint8]:
        # Make a copy of the current frame
        result_bgr = frame_bgr.copy()
        # Draw tracking
        for id, tracked_vehicles in tracked_vehicles.items():
            cv2.polylines(result_bgr, [np.stack(tracked_vehicles['centroid_history'])[:, None, :]], False, DRAW_BGR_COLOR_PALETTE[id % len(DRAW_BGR_COLOR_PALETTE)], DRAW_TRACKING_THICKNESS, cv2.LINE_8)
        # Draw boxes for all cars
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['vehicle_roi']
            cv2.rectangle(result_bgr, (x1, y1), (x2, y2), DRAW_BGR_WHITE, DRAW_BOX_THICKNESS)
        # Draw a different box for cars whose a plate was detected
        for plate in plates:
            x1, y1, x2, y2 = plate['vehicle_roi']
            u1, v1, u2, v2 = plate['plate_roi']
            width, height = TARGET_WIDTH // 2, TARGET_HEIGHT // 2
            result_bgr[y1:y1+height, x1:x1+width] = cv2.resize(plate['warped_plate'], (width, height))
            cv2.rectangle(result_bgr, (x1, y1), (x2, y2), DRAW_BGR_PASTEL_PINK, DRAW_BOX_THICKNESS)
            cv2.rectangle(result_bgr, (x1, y1), (x1+width, y1+height), DRAW_BGR_PASTEL_PINK, DRAW_BOX_THICKNESS)
            cv2.rectangle(result_bgr, (x1 + u1, y1 + v1), (x1 + u2, y1 + v2), DRAW_BGR_PASTEL_GREEN, DRAW_BOX_THICKNESS)
        # Return the resulting frame
        return result_bgr

    def _compute_binary_image_of_chars(self, warped_plate_bin: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # Find the contour of the objects in the binary image
        contours, _ = cv2.findContours(cv2.bitwise_not(warped_plate_bin), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Try to get the list of objects that behave like a character
        char_contours = list()
        for contour in contours:
            u1, v1 = contour.min(axis=(0, 1))
            u2, v2 = contour.max(axis=(0, 1))
            box_height, box_width = v2 - v1, u2 - u1
            box_area = box_width * box_height
            if box_width <= box_height and 0.02 <= (box_area / TARGET_AREA) <= 0.15:
                char_contours.append(contour)
        # Draw the objects that behave like a charactere 
        warped_chars_bin = np.full((TARGET_HEIGHT, TARGET_WIDTH), 255, dtype=np.uint8)
        cv2.drawContours(warped_chars_bin, char_contours, -1, 0, cv2.FILLED, cv2.LINE_4)
        # Return the final binary image
        return cv2.bitwise_or(warped_chars_bin, warped_plate_bin)

    def _convert_from_gray_to_binary(self, warped_plate_gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # Smooth the input image preserving edges
        warped_plate_blur = cv2.bilateralFilter(warped_plate_gray, 9, 75, 75)
        # Apply adaptive thresholding for binarization
        return cv2.adaptiveThreshold(warped_plate_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #
        #TODO Aplicar técnica baseada em deep learning para melhorar a imagem
        # dilated_img = cv2.dilate(warped_plate_gray, np.ones((7, 7), np.uint8))
        # bg_img = cv2.medianBlur(dilated_img, 21)
        # diff_img = 255 - cv2.absdiff(warped_plate_gray, bg_img)
        # norm_img = diff_img.copy()
        # cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
        # cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # os.makedirs(os.path.join('results', 'convert_from_gray_to_binary'), exist_ok=True)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'0-input_gray.png'), warped_plate_gray)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'1-dilated_img.png'), dilated_img)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'2-bg_img.png'), bg_img)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'3-diff_img.png'), diff_img)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'4-norm_img.png'), norm_img)
        # cv2.imwrite(os.path.join('results', 'convert_from_gray_to_binary', f'5-thr_img.png'), thr_img)
        # return cv2.adaptiveThreshold(thr_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def _warp_to_target_rectangle(self, corners: NDArray[np.int32], images: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        # Sort the input corners to match the sequence of target (rectangle) corners
        mu = corners.mean(axis=0)
        sigma = np.cov(corners.T)
        u, _, _ = np.linalg.svd(sigma)
        if u[0, 0] < 0: u[:, 0] *= -1
        if u[1, 1] < 0: u[:, 1] *= -1
        d = np.matmul(corners - mu, u)
        angle = np.arctan2(d[:, 1], d[:, 0])
        source_pts = np.asarray(corners[np.argsort(angle)], dtype=np.float32)
        # Compute the affine transformation that maps the input corners to the target corners
        affine_mtx = cv2.estimateAffine2D(source_pts, TARGET_CORNERS)[0]
        # Apply the affine transformation to input images
        return (cv2.warpAffine(image, affine_mtx, (TARGET_WIDTH, TARGET_HEIGHT)) for image in images)

    def _detect_plate(self, vehicle: DetectedVehicle, *, lpd_model: YOLO, ocr_model: Reader) -> List[DetectedPlate]:
        #TODO Fazer OCR em batch
        result: List[DetectedPlate] = list()
        # The vehicle's image may include several plate box candidates
        vehicle_bgr = vehicle['vehicle']
        for plate_box in lpd_model(vehicle_bgr, device=lpd_model.device, verbose=False)[0].boxes:
            # The plate box includes a perspective projection of the plate, so we have to find the quadrilateral resulting from this projection
            u1, v1, u2, v2 = map(lambda arg: arg.item(), plate_box.xyxy.round().to(torch.int32)[0].cpu())
            plate_bgr = vehicle_bgr[v1:v2, u1:u2]
            plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
            plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            plate_contours, _ = cv2.findContours(plate_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            plate_contours = sorted(plate_contours, key=cv2.contourArea, reverse=True)[:5]
            # Each contour is a candidate projected plate
            for plate_contour in plate_contours:
                # But the contours may be defined by more than four vertices, so we have to simplify it
                chull = cv2.convexHull(plate_contour, clockwise=False, returnPoints=True)
                while len(chull) > 4:
                    vertices = np.concatenate((np.concatenate((chull[-1][None, ...], chull, chull[0][None, ...]), axis=0).squeeze(), np.ones((len(chull) + 2, 1), dtype=chull.dtype)), axis=1)
                    matrices = np.stack((vertices[:-2], vertices[1:-1], vertices[2:]), axis=1, dtype=np.float32)
                    min_idx = np.argmin(np.abs(np.linalg.det(matrices)))
                    chull = np.delete(chull, min_idx, axis=0)
                # Given a convex quadrilateral...
                if len(chull) == 4:
                    # ... warp it to the rectangular shape of the licence plate, ...
                    warped_plate_bgr, warped_plate_gray = self._warp_to_target_rectangle(chull.squeeze(), (plate_bgr, plate_gray))
                    # ... convert the warped gray image of the plate to a binary image, ...
                    warped_plate_bin = self._convert_from_gray_to_binary(warped_plate_gray)
                    # ... compute an binary image having only the characters of the plate, ...
                    warped_chars_bin = self._compute_binary_image_of_chars(warped_plate_bin)
                    # ... and perform OCR
                    plate = ''.join(ocr_model.readtext(warped_chars_bin, detail=0, allowlist=CHAR_WHITELIST))
                    if len(plate) >= 5:
                        result.append(DetectedPlate(
                            plate=plate,
                            warped_plate=warped_plate_bgr,
                            warped_plate_ocr=warped_chars_bin,
                            vehicle=vehicle_bgr,
                            vehicle_roi=vehicle['vehicle_roi'],
                            plate_roi=(u1, v1, u2, v2),
                            when=vehicle['when'],
                            tracked_vehicle=vehicle['tracked_vehicle'],
                        ))
        return result
    
    def _make_reported_plate(self, tracked_vehicle: TrackedVehicle) -> ReportedPlate:
        history = tracked_vehicle['plate_history']
        if len(history) == 1:
            # Peek the single plate we have
            best_detected_plate = history[0]
            plate = best_detected_plate['plate']
        else:
            # Check the frequency of each character in each entry of the plate
            num_entries = max(map(lambda arg: len(arg['plate']), history))
            count = np.zeros((num_entries, len(CHAR_WHITELIST)), dtype=np.uint32)
            for detected_plate in history:
                for entry_idx, entry in enumerate(detected_plate['plate'].upper()):
                    count[entry_idx, CHAR_WHITELIST.index(entry)] += 1
            # Keep the most voted characters
            winner_idx = count.argmax(axis=1)
            plate = ''.join(map(lambda idx: CHAR_WHITELIST[idx], winner_idx))
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
            id=tracked_vehicle['id'],
            plate=plate,
            vehicle=best_detected_plate['vehicle'],
            warped_plate=best_detected_plate['warped_plate'],
            warped_plate_ocr=best_detected_plate['warped_plate_ocr'],
            when=best_detected_plate['when'],
        )

    def _perform_tracking(self, boxes: Boxes, now: datetime, frame_size: Tuple[int, int], tracker_model: BYTETracker, tracked_vehicles: Dict[int, TrackedVehicle], patience_in_seconds: float, include_debug_info: bool) -> Tuple[Dict[ROI, TrackedVehicle], Set[ROI], List[TrackedVehicle]]:
        # Create resulting lists
        online: Dict[ROI, TrackedVehicle] = dict()
        created: Set[TrackedVehicle] = set()
        deleted: List[TrackedVehicle] = list()
        # Update the tracker with the detected boxes
        online_vehicles = tracker_model.update(boxes.data.cpu(), None)
        for data in online_vehicles:
            x1, y1, x2, y2, id, *_ = map(lambda arg: arg.item(), data.round().astype(np.int32))
            vehicle_roi = (min(max(x1, 0), frame_size[0] - 1), min(max(y1, 0), frame_size[1] - 1), min(max(x2, 0), frame_size[0] - 1), min(max(y2, 0), frame_size[1] - 1))
            if (vehicle_roi[2] - vehicle_roi[0]) > 0 and (vehicle_roi[3] - vehicle_roi[1]) > 0:
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                tracked_vehicle = tracked_vehicles.get(id, None)
                # Check whether the current tracked vehicle is new and report this vehicle as "created"
                if tracked_vehicle is None:
                    tracked_vehicle = TrackedVehicle(
                        id=id,
                        last_seen=now,
                        **(dict(centroid_history=[centroid]) if include_debug_info else dict()),
                        plate_history=[],
                    )
                    created.add(vehicle_roi)
                    tracked_vehicles[id] = tracked_vehicle
                # Otherwise, update its information
                else:
                    tracked_vehicle['last_seen'] = now
                    if include_debug_info:
                        tracked_vehicle['centroid_history'].append(centroid)
                online[vehicle_roi] = tracked_vehicle
        # Remove vehicles that have not been seen for a long time
        for id, tracked_vehicle in tracked_vehicles.items():
            elapsed_time = (now - tracked_vehicle['last_seen']).total_seconds()
            if elapsed_time > patience_in_seconds:
                deleted.append(tracked_vehicle)
        for tracked_vehicle in deleted:
            del tracked_vehicles[tracked_vehicle['id']]
        # Return the list of current vehicles, new vehicles, and vehicles that are missing for a long time
        return online, created, deleted
    
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
                            storage_task = StorageTask(data['task'])
                            filename = data['filename']
                            if storage_task == StorageTask.CREATE:
                                if not os.path.exists(os.path.join(self._final_storage_dir, filename)):
                                    self._send(server_address, AgentEvent.DETECTION_CREATE, data)
                                    os.rename(os.path.join(self._local_storage_dir, filename), os.path.join(self._final_storage_dir, filename))
                                else:
                                    os.remove(os.path.join(self._local_storage_dir, filename))
                                os.remove(os.path.join(self._local_storage_dir, json_filename))
                            elif storage_task == StorageTask.UPDATE:
                                self._send(server_address, AgentEvent.DETECTION_UPDATE, data)
                                if os.path.exists(os.path.join(self._final_storage_dir, filename)):
                                    os.remove(os.path.join(self._final_storage_dir, filename))
                                os.rename(os.path.join(self._local_storage_dir, filename), os.path.join(self._final_storage_dir, filename))
                                os.remove(os.path.join(self._local_storage_dir, json_filename))
                            else:
                                raise NotImplementedError
                except Exception as error:
                    self._logger.error(f'{self.tag_slug}\tfinal storage worker - exception handler, loop still running: "{str(error)}"')
                    time.sleep(10)
        except:
            self._logger.exception(f'{self.tag_slug}\tfinal storage worker - end by exception')
            raise            
        self._logger.debug(f'{self.tag_slug}\tfinal storage worker - end')

    def _run_local_storage_worker(self, worker_idx: int, storage_queue: "Queue[Tuple[StorageTask, ReportedPlate]]") -> None:
        PASSWORD_CHARS = string.ascii_letters + string.digits
        PASSWORD_SIZE = 8
        self._logger.debug(f'{self.tag_slug}\tlocal storage worker #{worker_idx} - begin')
        try:
            # While we have data to store...
            data = storage_queue.get()
            while data is not None:
                storage_task, reported_plate = data
                try:
                    # Encode the plate as a PNG file
                    success, vehicle_png = cv2.imencode('.png', reported_plate['vehicle'])
                    if not success: raise Exception('Error converting to PNG')
                    success, warped_plate_png = cv2.imencode('.png', reported_plate['warped_plate'])
                    if not success: raise Exception('Error converting to PNG')
                    success, warped_plate_ocr_png = cv2.imencode('.png', reported_plate['warped_plate_ocr'])
                    if not success: raise Exception('Error converting to PNG')
                    with self._local_storage_lock:
                        # Put all information in a ZIP file
                        filename = f'{reported_plate["id"]}.zip'
                        password = ''.join(random.choice(PASSWORD_CHARS) for _ in range(PASSWORD_SIZE))
                        info = StoredPlate(
                            plate=reported_plate['plate'],
                            when=reported_plate['when'].isoformat(sep=' '),
                        )
                        with ZipFile(os.path.join(self._local_storage_dir, filename), mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                            zf.writestr('info.json', json.dumps(info))
                            zf.writestr('vehicle.png', vehicle_png.tostring())
                            zf.writestr('warped_plate.png', warped_plate_png.tostring())
                            zf.writestr('warped_plate_ocr.png', warped_plate_ocr_png.tostring())
                        with open(os.path.join(self._local_storage_dir, filename), mode='rb') as fp:
                            md5sum = hashlib.md5()
                            for buf in iter(partial(fp.read, 128), b''):
                                md5sum.update(buf)
                        # Write a header to a JSON file
                        if storage_task == StorageTask.UPDATE or (storage_task == StorageTask.CREATE and not os.path.exists(os.path.join(self._local_storage_dir, f'{filename}.json'))):
                            with open(os.path.join(self._local_storage_dir, f'{filename}.json'), 'w') as fp:
                                json.dump({
                                    'task': storage_task,
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

    def _run_vehicle_worker(self, worker_idx: int, lpd_model: YOLO, ocr_model: Reader, vehicle_queue: "Queue[DetectedVehicle]", detected_plates: List[DetectedPlate]) -> None:
        # Get first vehicle
        detected_vehicle = vehicle_queue.get()
        try:
            # If there is a vehicle to be processed then...
            while detected_vehicle is not None:
                vehicle_queue.task_done()
                # .... find each plate candidate in the vehicle's image...
                plate_candidates = self._detect_plate(detected_vehicle, lpd_model=lpd_model, ocr_model=ocr_model)
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
        
    def _run_video_loop(self, server_address: str, video_uri: str, output_filename: Optional[str], output_fourcc: Optional[str], tracking_patience_in_seconds: float, num_vehicle_workers: int, num_local_storage_workers: int, vehicle_queue_size: int, storage_queue_size: int) -> None:
        draw_debug_info = output_filename is not None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        self._status |= AgentStatus.RUNNING
        self._logger.debug(f'{self.tag_slug}\tprocessing thread - begin (using "{device}")')
        try:
            # Get information about the video
            try:
                video = VideoCapture(video_uri)
                fourcc = cv2.VideoWriter_fourcc(*output_fourcc) if output_fourcc is not None else int(video.get(cv2.CAP_PROP_FOURCC))
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_rate = video.get(cv2.CAP_PROP_FPS)
                frame_size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                video.release()
            # Create a YOLO model pre-trained on the COCO dataset and find the ID of the vehicle classes
            coco_model = YOLO(os.path.join(config_dir, 'yolov8n.pt')).to(device)
            coco_ids = list(coco_model.names.keys())
            coco_names = list(coco_model.names.values())
            used_coco_ids = [coco_ids[coco_names.index(name)] for name in USED_COCO_CLASS_NAMES]
            # Create a tracker model
            tracker_model = BYTETracker(track_thresh=0.0, track_buffer=int(tracking_patience_in_seconds * frame_rate + 0.5), match_thresh=0.95, frame_rate=int(frame_rate))  # track_thresh=0.0 means we want to keep all boxes
            tracked_vehicles: Dict[int, TrackedVehicle] = dict()
            # Create a set of YOLO models pre-trained on a LPR dataset (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and a set of EasyOCR models
            lpd_models = [YOLO(os.path.join(config_dir, 'yolov8lpd.pt')).to(device) for _ in range(num_vehicle_workers)]
            ocr_models = [Reader(['en'], gpu=torch.cuda.is_available()) for _ in range(num_vehicle_workers)]
            # Create the queues that keep the data to be processed and stored locally
            vehicle_queue: "Queue[DetectedVehicle]" = Queue(maxsize=vehicle_queue_size)
            storage_queue: "Queue[Tuple[StorageTask, ReportedPlate]]" = Queue(maxsize=storage_queue_size)
            # Create and start local storage workers
            local_storage_workers = [threading.Thread(target=self._run_local_storage_worker, args=(worker_idx, storage_queue,), daemon=True) for worker_idx in range(num_local_storage_workers)]
            for worker in local_storage_workers: worker.start()
            # Create and start the final storage worker
            final_storage_worker = threading.Thread(target=self._run_final_storage_worker, args=(server_address,), daemon=True)
            final_storage_worker.start()
            # Set a loop to restart the video processing in case of some exception
            while self._keep_running:
                try:
                    # Open the video stream
                    self._status |= AgentStatus.PROCESSING 
                    self._send(server_address, AgentEvent.STREAM_OPEN, None)
                    self._logger.info(f'{self.tag_slug}\tstream opened')
                    try:
                        video_writer = VideoWriter(output_filename, fourcc, frame_rate, frame_size) if draw_debug_info else None
                        try:
                            # For each frame...
                            for vehicles in tqdm(coco_model(video_uri, classes=used_coco_ids, device=coco_model.device, stream=True, stream_buffer=draw_debug_info, verbose=False), desc=f'Processing "{video_uri}"', total=frame_count, disable=(not os.path.isfile(video_uri))):
                                if not self._keep_running:
                                    break
                                now = datetime.now()
                                frame_bgr = vehicles.orig_img
                                # ... update the tracking system, ...
                                online, created, deleted = self._perform_tracking(vehicles.boxes, now, frame_size, tracker_model, tracked_vehicles, tracking_patience_in_seconds, draw_debug_info)
                                # ... get each detected vehicle and enqueue them for processing, ...
                                detected_vehicles: List[DetectedVehicle] = list()
                                for vehicle_roi, tracked_vehicle in online.items():
                                    detected_vehicle = DetectedVehicle(
                                        tracked_vehicle=tracked_vehicle,
                                        vehicle=frame_bgr[vehicle_roi[1]:vehicle_roi[3], vehicle_roi[0]:vehicle_roi[2]],
                                        vehicle_roi=vehicle_roi,
                                        when=now,
                                    )
                                    detected_vehicles.append(detected_vehicle)
                                    vehicle_queue.put(detected_vehicle)
                                for _ in range(num_vehicle_workers): vehicle_queue.put(None)  # Enqueue empty data to sinalize the end of processing
                                # ... create and start vehicle processing workers using YOLO models pre-trained on a LPR dataset (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4), ...
                                detected_plates: List[DetectedPlate] = list()
                                vehicle_workers = [threading.Thread(target=self._run_vehicle_worker, args=(worker_idx, lpd_models[worker_idx], ocr_models[worker_idx], vehicle_queue, detected_plates,), daemon=True) for worker_idx in range(num_vehicle_workers)]
                                for worker in vehicle_workers: worker.start()
                                vehicle_queue.join()
                                # ... assign detected licence plates to tracked vehicles, ...
                                for detected_plate in detected_plates:
                                    vehicle_roi = detected_plate['vehicle_roi']
                                    tracked_vehicle = detected_plate['tracked_vehicle']
                                    tracked_vehicle['plate_history'].append(detected_plate)
                                    if vehicle_roi in created:
                                        storage_queue.put((StorageTask.CREATE, self._make_reported_plate(tracked_vehicle)))
                                for tracked_vehicle in deleted:
                                    if len(tracked_vehicle['plate_history']) != 0:
                                        storage_queue.put((StorageTask.UPDATE, self._make_reported_plate(tracked_vehicle)))
                                # ... and draw results if needed
                                if video_writer is not None:
                                    video_writer.write(self._draw_debug_info(frame_bgr, detected_vehicles, detected_plates, tracked_vehicles))
                            self._logger.info(f'{self.tag_slug}\tstream closed')
                            if video_writer is not None:
                                self._keep_running = False
                        finally:
                            if video_writer is not None:
                                video_writer.release()
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

    def _send(self, server_address: str, event: AgentEvent, data: Optional[Dict[str, Any]]) -> None:
        self._logger.debug(f'{self.tag_slug}\tsend - begin')
        try:
            #TODO print(f'licenceplate agent "{self.tag_slug}" reported the event "{event}" and sent the following data to "{server_address}" -> {data}')  # Ticket
            pass
        except:
            self._logger.exception(f'{self.tag_slug}\tsend - end by exception')
            raise
        self._logger.debug(f'{self.tag_slug}\tsend - end')

    def run(self, server_address: str, video_uri: str, output_filename: Optional[str] = default.OUTPUT_FILENAME, output_fourcc: Optional[str] = default.OUTPUT_FOURCC, tracking_patience_in_seconds: float = default.TRACKING_PATIENCE_IN_SECONDS, num_vehicle_workers: int = default.NUM_VEHICLE_WORKERS, num_local_storage_workers: int = default.NUM_LOCAL_STORAGE_WORKERS, vehicle_queue_size: int = default.VEHICLE_QUEUE_SIZE, storage_queue_size: int = default.STORAGE_QUEUE_SIZE) -> None:
        self._logger.info(f'{self.tag_slug}\trun - begin [video_uri="{video_uri}", num_vehicle_workers={num_vehicle_workers}, num_local_storage_workers={num_local_storage_workers}, vehicle_queue_size={vehicle_queue_size}, storage_queue_size={storage_queue_size}]')
        try:
            # Check if the agent is already running
            if AgentStatus.RUNNING in self.status: raise Exception('The agent is running already')
            self._keep_running = True
            try:
                # Start the main thread of the agent
                thread = threading.Thread(target=self._run_video_loop, args=(server_address, video_uri, output_filename, output_fourcc, tracking_patience_in_seconds, num_vehicle_workers, num_local_storage_workers, vehicle_queue_size, storage_queue_size,), daemon=True)
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
