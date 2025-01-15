from typing import Any, Final, Optional
import time


class ElapsedTime():

    def __init__(self, name: str = 'Elapsed Time', verbose: bool = True):
        self._begin: Optional[float] = None
        self._end: Optional[float] = None
        self.name: Final = name
        self.verbose: Final = verbose

    def __enter__(self) -> "ElapsedTime":
        self._begin = time.perf_counter()
        return self

    def __exit__(self, *args: Any, **kwargs: Any):
        self._end = time.perf_counter()
        if self.verbose:
            self.print()

    def print(self) -> None:
        print(f'{self.name}: {self._end - self._begin:1.4f} seconds')
