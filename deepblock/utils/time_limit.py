"""
From: https://stackoverflow.com/a/601168/16407115

Usage:
```python
try:
    with time_limit(10):
        long_function_call()
except TimeoutException as e:
    print("Timed out!")
```
"""

import signal
import threading
import _thread
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit_unix(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@contextmanager
def time_limit(seconds):
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out!")
    finally:
        timer.cancel()
