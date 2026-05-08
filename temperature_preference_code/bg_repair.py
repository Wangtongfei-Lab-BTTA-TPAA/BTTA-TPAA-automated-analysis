"""
Module Name: bg_repair
Affiliation: CIBR_WangtongfeiLab

Description:
    Interactively repair visible-light or infrared background images. The user
    selects a circular region in an existing background image, chooses a video
    frame as the replacement source, and blends the selected region into the
    saved background image.

Functions:
    - parse_time_string: Convert MM:SS or HH:MM:SS input to seconds.
    - main: Launch the interactive background repair tool.

Usage:
    python bg_repair.py

Copyright:
    Copyright (c) CIBR_WangtongfeiLab.

Note:
    If this code is used in academic publication, please cite or acknowledge
    CIBR_WangtongfeiLab.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


KEY_ENTER = 13
KEY_SAVE = 115
KEY_FORWARD = 120
KEY_BACKWARD = 122


def parse_time_string(value):
    """Convert MM:SS or HH:MM:SS input to seconds.

    Args:
        value: User-provided time string.

    Returns:
        Integer seconds, or None if the input cannot be parsed.
    """
    try:
        parts = value.strip().replace('：', ':').split(':')
        if len(parts) == 2:
            return 60 * int(parts[0]) + int(parts[1])
        if len(parts) == 3:
            return 3600 * int(parts[0]) + 60 * int(parts[1]) + int(parts[2])
    except ValueError:
        return None
    return None


class BgRepair:
    """Interactive background image repair workflow."""

    def __init__(self):
        """Initialize selected center point and repair radius."""
        self.ix = -1
        self.iy = -1
        self.ir = 25
        self.fps = 0
        self.frames_nb = 0

    def callback(self, event, x, y, flags, param):
        """Store the latest mouse position in the OpenCV window."""
        self.ix = x
        self.iy = y

    def run(self):
        """Run the interactive repair workflow."""
        video_paths = [cfg['video_path'], cfg['video_path2']]
        bg_names = ['bg.bmp', 'bg_ir.bmp']
        bg_dir = Path(video_paths[0]).parent / Path(video_paths[0]).stem

        selected_video = self._select_video(video_paths)
        bg_path = bg_dir / bg_names[selected_video - 1]
        bg = cv2.imread(str(bg_path))
        cv2.imshow('bg', bg)

        self._select_repair_region(bg)
        mask = self._build_region_mask(bg)

        video_path = video_paths[selected_video - 1]
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frames_nb = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        secs = self._prompt_time_point()

        frame = self._select_source_frame(cap, secs, bg_path)
        if frame is None:
            return

        out = self._blend_region(bg, frame, mask)
        self.save(bg_path, out)

    def _select_video(self, video_paths):
        """Prompt the user to select the visible-light or infrared video."""
        print('Please select the background image of the video to process:')
        print(f'1 {video_paths[0]}')
        print(f'2 {video_paths[1]}')
        print('Please enter 1 or 2:')
        while True:
            try:
                selected = int(input().strip())
            except ValueError:
                print('Invalid value. Please input again:')
                continue
            if selected in [1, 2]:
                return selected
            print('Invalid value. Please input again:')

    def _select_repair_region(self, bg):
        """Let the user choose the circular repair region."""
        print('\n-----------------------------')
        print('Operating Instructions:')
        print('enter: confirm')
        print('z: increase the radius')
        print('x: decrease the radius')
        while True:
            cv2.setMouseCallback('bg', self.callback)
            key = cv2.waitKeyEx(20)
            if key == KEY_ENTER:
                break
            if key == KEY_BACKWARD:
                self.ir += 1
            elif key == KEY_FORWARD:
                self.ir -= 1

            if self.ix >= 0 and self.iy >= 0:
                img = bg.copy()
                cv2.circle(img, (self.ix, self.iy), self.ir, (0, 0, 255), 1)
                cv2.circle(img, (self.ix, self.iy), self.ir + 1, (255, 255, 255), 1)
                cv2.imshow('bg', img)

    def _build_region_mask(self, bg):
        """Build and preview the selected circular mask."""
        mask = np.zeros_like(bg)
        cv2.circle(mask, (self.ix, self.iy), self.ir, (255, 255, 255), -1)
        cv2.imshow('mask', mask)
        cv2.waitKeyEx(700)
        cv2.destroyAllWindows()
        return mask

    def _prompt_time_point(self):
        """Prompt the user for the source frame time point."""
        print('\n-----------------------------')
        print('Please enter a time point (for example: 12:30 or 1:10:25):')
        while True:
            secs = parse_time_string(input())
            if secs is not None:
                return secs
            print('Invalid value. Please input again:')

    def _select_source_frame(self, cap, secs, bg_path):
        """Let the user move through video frames and select a source frame."""
        nowp = int(secs * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, nowp)
        _, frame = cap.read()
        dst = frame.copy()
        self._draw_selected_region(frame)

        print('\n-----------------------------')
        print('Operating Instructions:')
        print('enter: confirm')
        print('z: backward')
        print('x: forward')
        print('s: save this frame as background image')
        while True:
            cv2.imshow('video', frame)
            key = cv2.waitKeyEx(10)
            if key == KEY_FORWARD:
                _, frame = cap.read()
                dst = frame.copy()
                self._draw_selected_region(frame)
                nowp += 1
                nowp = self.frames_nb - 1 if nowp > self.frames_nb else nowp
            elif key == KEY_BACKWARD:
                nowp = max(nowp - 10, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, nowp)
                _, frame = cap.read()
                dst = frame.copy()
                self._draw_selected_region(frame)
            elif key == KEY_SAVE:
                self.save(bg_path, dst)
                return None
            elif key == KEY_ENTER:
                break
            print(f'\r{nowp}/{self.frames_nb}', end='')
        return dst

    def _draw_selected_region(self, frame):
        """Draw the selected repair circle on a frame."""
        cv2.circle(frame, (self.ix, self.iy), self.ir, (0, 0, 255), 1)
        cv2.circle(frame, (self.ix, self.iy), self.ir + 1, (255, 255, 255), 1)

    def _blend_region(self, bg, frame, mask):
        """Blend the selected frame region into the background image."""
        mask = mask.astype(bool)
        mask_inverse = ~mask
        dst = frame * mask
        cv2.imshow('dst', dst)
        cv2.waitKeyEx(700)
        cv2.destroyAllWindows()

        bg_roi = (bg * mask).astype(float)
        dst = dst.astype(float)
        out = bg.copy()
        for i in range(100):
            ratio = (i + 1) / 100
            roi = np.round(bg_roi * (1 - ratio) + dst * ratio).astype(np.uint8)
            out = mask_inverse * bg + roi
            cv2.imshow('out', out)
            cv2.waitKeyEx(10)
        print('\n\nA new background image has been calculated.')
        cv2.waitKeyEx(700)
        return out

    def save(self, bg_path, out):
        """Backup the old background and save the repaired image."""
        old_bg_path = bg_path.parent / f'{bg_path.stem}_old.bmp'
        shutil.copy(bg_path, old_bg_path)
        cv2.imwrite(str(bg_path), out)
        print(f'Have saved it to {bg_path}\nexit.')


def main():
    """Launch the background repair workflow."""
    repair = BgRepair()
    repair.run()


if __name__ == '__main__':
    main()
