import time


class Timer:

    def __init__(self, start_now=False) -> None:
        self.__start_time = time.time() if start_now else None

    def start(self):
        self.__start_time = time.time()

    def lap(self):
        lap = time.time() - self.__start_time
        self.__start_time = time.time()
        return lap
