from typing import Final


OUTPUT_FILENAME: Final[str] = None
OUTPUT_FOURCC: Final[str] = None  # `None`` means that it will be taken from the input video

TRACKING_PATIENCE_IN_SECONDS: Final[float] = 3.0

NUM_LOCAL_STORAGE_WORKERS: Final[int] = 2
STORAGE_QUEUE_SIZE: Final[int] = 300

NUM_VEHICLE_WORKERS: Final[int] = 4
VEHICLE_QUEUE_SIZE: Final[int] = 300
