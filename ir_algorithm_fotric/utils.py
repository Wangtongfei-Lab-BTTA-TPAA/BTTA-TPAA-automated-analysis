#! python
"""
Module Name: utils
Affiliation: CIBR_WangtongfeiLab

Description:
    Provide shared utility helpers for configuration loading, video discovery,
    progress display, and simple console waiting animation.

Functions:
    - load_cfg: Load YAML configuration into an attribute-access dictionary.
    - get_temp_folder_name: Generate a short random temporary folder name.
    - get_video_paths: Collect configured input video paths.
    - Pbar: Print a simple terminal progress bar.
    - Wait: Print a simple waiting animation around a code block.

Usage:
    Import helpers from processing and analysis scripts.

Copyright:
    Copyright (c) CIBR_WangtongfeiLab.

Note:
    If this code is used in academic publication,
    please cite or acknowledge CIBR_WangtongfeiLab.
"""

import random
import time
import yaml
from multiprocessing import Process
from pathlib import Path


class EasyDict(dict):
    """Dictionary with recursive attribute access for configuration values."""

    def __init__(self, d=None, **kwargs):
        """Initialize from a dictionary and optional keyword overrides."""
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)

        for key, value in d.items():
            setattr(self, key, value)

        for key in self.__class__.__dict__.keys():
            if not (key.startswith('__') and key.endswith('__')) and key not in ('update', 'pop'):
                setattr(self, key, getattr(self, key))

    def __setattr__(self, name, value):
        """Set attributes and dictionary entries together."""
        if isinstance(value, (list, tuple)):
            value = [self.__class__(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)

        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        """Update values while preserving attribute access."""
        data = e or dict()
        data.update(f)
        for key in data:
            setattr(self, key, data[key])

    def pop(self, key, default=None):
        """Remove a value from both attribute and dictionary access."""
        delattr(self, key)
        return super(EasyDict, self).pop(key, default)


def load_cfg(path):
    """Load a YAML config file and convert literal 'None' strings to None."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        if value == 'None':
            cfg[key] = None

    return EasyDict(cfg)


def __get_random_str(length=5):
    """Generate a random alphabetic string."""
    elements = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return ''.join(random.choices(elements, k=length))


def get_temp_folder_name():
    """Generate an eight-character temporary folder name."""
    return __get_random_str(length=8)


def get_video_paths(cfg):
    """Collect input videos from the configured file or folder path."""
    data_path = Path(cfg.Data_path)
    if data_path.is_file():
        if data_path.suffix in cfg.Fmts:
            videos = [str(data_path)]
        else:
            print(f'error: the file not end with {cfg.Fmts}')
            videos = []
    else:
        videos = [path for path in data_path.iterdir() if path.suffix in cfg.Fmts]

    return [str(video) for video in videos]


class Pbar:
    """Simple terminal progress bar with elapsed and estimated remaining time."""

    def __init__(self, total, pbar_len=50, pbar_value='|', pbar_blank='-'):
        """Initialize progress state."""
        self.total = total
        self.pbar_len = pbar_len
        self.pbar_value = pbar_value
        self.pbar_blank = pbar_blank
        self.now = 0
        self.time = time.time()
        self.start_time = time.time()
        self.close_flag = False

    def update(self, nb=1, set=False, set_value=None):
        """Advance or set progress and print the current progress bar."""
        if set:
            self.now = set_value
        else:
            self.now += nb

        percent = int(round(self.now / self.total * 100))
        pbar_now = round(self.pbar_len * percent / 100)
        if pbar_now > self.pbar_len:
            pbar_now = self.pbar_len

        blank_len = self.pbar_len - pbar_now
        time_used = time.time() - self.time
        speed = nb / (time_used + 1e-4)
        total_time_used = time.time() - self.start_time
        total_time_used_min, total_time_used_sec = divmod(total_time_used, 60)
        total_time_used = f'{int(total_time_used_min):0>2d}:{int(total_time_used_sec):0>2d}'

        remaining_it = self.total - self.now if self.total - self.now >= 0 else 0
        remaining_time = remaining_it / speed
        remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)
        remaining_time = f'{int(remaining_time_min):0>2d}:{int(remaining_time_sec):0>2d}'

        pbar = f'{percent:>3d}%|{self.pbar_value * pbar_now}{self.pbar_blank * blank_len}| '
        pbar += f'{self.now}/{self.total} [{total_time_used}<{remaining_time}, {speed:.2f}it/s]'
        print(f'\r{pbar}', end='')
        self.time = time.time()

    def close(self, reset_done=True):
        """Finish the progress bar and move output to the next line."""
        if self.close_flag:
            return
        if reset_done:
            self.update(set=True, set_value=self.total)
        print()
        self.close_flag = True


class Wait:
    """Simple context manager for printing a waiting animation."""

    def __init__(self, info=None):
        """Initialize display text for the waiting animation."""
        if info:
            self.info = f' {info}'
            print(f'0:00:00{self.info}.../', end='', flush=True)
        else:
            self.info = ' '
            print('0:00:00 .../', end='', flush=True)

    def __enter__(self):
        """Start the waiting animation process."""
        self.t0 = time.time()
        self.p = Process(target=self.print_fn, args=())
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the waiting animation process."""
        self.p.terminate()
        print('\bDone')

    def s_to_time(self, seconds):
        """Format seconds as H:MM:SS."""
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        hrs, mins, secs = int(round(hrs)), int(round(mins)), int(round(secs))
        return f'{hrs}:{mins:0>2}:{secs:0>2}'

    def print_fn(self):
        """Print the waiting animation until the process is terminated."""
        while True:
            time.sleep(1)
            print(f'\r{self.s_to_time(time.time() - self.t0)}{self.info}...\\', end='', flush=True)
            time.sleep(1)
            print(f'\r{self.s_to_time(time.time() - self.t0)}{self.info}.../', end='', flush=True)


if __name__ == '__main__':
    print(get_temp_folder_name())
    print(get_temp_folder_name())
    print(get_temp_folder_name())
