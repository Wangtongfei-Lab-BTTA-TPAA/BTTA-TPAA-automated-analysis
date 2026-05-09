"""
Module Name: utils
Affiliation: CIBR_WangtongfeiLab

Description:
    Provide small utility helpers used by the temperature-preference video
    processing scripts.

Functions:
    - stop_thread: Raise SystemExit in a target thread.

Usage:
    Imported by main.py and related scripts.

Copyright:
    Copyright (c) CIBR_WangtongfeiLab.

Note:
    If this code is used in academic publication, please cite or acknowledge
    CIBR_WangtongfeiLab.
"""

import ctypes
import inspect
import time


class Pbar:
    """Small terminal progress bar used by the video processing scripts."""

    def __init__(self, total, pbar_len=50, pbar_value='|', pbar_blank='-'):
        """Initialize progress bar state."""
        self.total = total
        self.pbar_len = pbar_len
        self.pbar_value = pbar_value
        self.pbar_blank = pbar_blank
        self.now = 0
        self.time = time.time()
        self.start_time = time.time()

    def update(self, nb=1, set=False, set_value=None):
        """Advance or set the progress bar and print the current status."""
        if set:
            self.now = set_value
        else:
            self.now += nb

        percent = round(self.now / self.total * 100)
        pbar_now = round(self.pbar_len * percent / 100)
        pbar_now = min(pbar_now, self.pbar_len)
        blank_len = self.pbar_len - pbar_now

        time_used = time.time() - self.time
        speed = nb / (time_used + 1e-4)
        total_time_used = self._format_seconds(time.time() - self.start_time)
        remaining_it = max(self.total - self.now, 0)
        remaining_time = self._format_seconds(remaining_it / speed)

        pbar = (
            f'{percent:>3d}%|{self.pbar_value * pbar_now}'
            f'{self.pbar_blank * blank_len}| '
            f'{self.now}/{self.total} '
            f'[{total_time_used}<{remaining_time}, {speed:.2f}it/s]'
        )
        print(f'\r{pbar}', end='')
        self.time = time.time()

    def close(self):
        """Finish the progress bar line."""
        print()

    @staticmethod
    def _format_seconds(seconds):
        """Format seconds as MM:SS."""
        minutes, seconds = divmod(seconds, 60)
        return f'{int(minutes):0>2d}:{int(seconds):0>2d}'


def stop_thread(thread):
    """Raise SystemExit in a thread."""
    tid = ctypes.c_long(thread.ident)
    exctype = SystemExit
    if not inspect.isclass(exctype):
        exctype = type(exctype)

    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if result == 0:
        raise ValueError('invalid thread id')
    if result != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError('PyThreadState_SetAsyncExc failed')
