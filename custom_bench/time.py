from typing import Type, Dict
import numpy as np
from time import sleep
from contextlib import contextmanager
from codetiming import Timer

timing_raw = {}


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


with _timer("test", timing_raw):
    np.random.randn(100000)
    sleep(1)

with _timer("test2", timing_raw):
    np.random.randn(100000)
    sleep(0.5)

print(timing_raw)
