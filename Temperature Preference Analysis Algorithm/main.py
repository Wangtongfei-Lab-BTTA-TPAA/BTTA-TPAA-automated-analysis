"""
Module Name: main
Affiliation: CIBR_WangtongfeiLab

Description:
    Process paired visible-light and infrared temperature-preference videos.
    The script segments mouse positions from the visible-light video, maps
    positions to the infrared video, estimates mouse and local background
    temperatures, and exports per-frame results.

Functions:
    - parse_time_value: Convert config time values to seconds.
    - build_runways: Build runway geometry objects from config coordinates.
    - videos_have_same_frame_count: Validate visible/infrared video alignment.
    - main: Load configuration and run the analysis workflow.

Usage:
    python main.py

Copyright:
    Copyright (c) CIBR_WangtongfeiLab.

Note:
    If this code is used in academic publication, please cite or acknowledge
    CIBR_WangtongfeiLab.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from utils import Pbar


with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

BAR_ROI = cfg['bar_roi']
REGION_NUMS = cfg['region_nums']
REGION_TH = cfg['region_th']
TH = cfg['th']
OUTLIER = tuple(cfg['outlier'])

SMALL_RECT_L = cfg['small_rect_l']
if SMALL_RECT_L is None or str(SMALL_RECT_L).lower() == 'none':
    SMALL_RECT_L = None

RUNWAY_COLOR = (255, 230, 255)
SPACE_KEY = 32
TRACK_POINTS_FILE = 'track_pss.npy'


def parse_time_value(value):
    """Convert a config time value to seconds.

    Args:
        value: Integer seconds, "None", or a string formatted as MM:SS or
            HH:MM:SS.

    Returns:
        The number of seconds, or None when the value is "None".
    """
    if isinstance(value, int):
        return value
    if value == 'None':
        return None

    parts = list(map(int, str(value).split(':')))
    units = [3600, 60, 1][-len(parts):]
    return sum(part * unit for part, unit in zip(parts, units))


def interpolate_pair_between_points(p1, p2, low_ratio, high_ratio):
    """Return two interpolated points between two points."""
    near_p1 = (
        p1[0] + low_ratio * (p2[0] - p1[0]),
        p1[1] + low_ratio * (p2[1] - p1[1]),
    )
    near_p2 = (
        p1[0] + high_ratio * (p2[0] - p1[0]),
        p1[1] + high_ratio * (p2[1] - p1[1]),
    )
    return near_p1, near_p2


def distance_between_points(a, b):
    """Return Euclidean distance between two 2D points."""
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def bgr_weighted_distance(pixel, palette):
    """Return weighted BGR color distance from one pixel to a palette."""
    b1, g1, r1 = pixel
    b2, g2, r2 = palette[:, 0], palette[:, 1], palette[:, 2]
    rm = (r1 + r2) / 2
    b = b1 - b2
    g = g1 - g2
    r = r1 - r2
    return (2 + rm / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rm) / 256) * (b ** 2)


def color_distance_to_palette(bgrs, color_bar):
    """Return pairwise weighted BGR distances from color-bar colors to pixels."""
    return np.stack([bgr_weighted_distance(color, bgrs) for color in color_bar], axis=-1)


def build_runways():
    """Build runway objects from the configured runway point sets."""
    runway_params = [
        [[tuple(point) for point in cfg[ps_key]], RUNWAY_COLOR]
        for ps_key in cfg['ps']
    ]
    return [Runway(*params) for params in runway_params]


class Runway:
    """Geometry and drawing helpers for one runway."""

    def __init__(
            self,
            ps4,
            col=(255, 255, 255),
            region_nums=REGION_NUMS,
    ):
        """Initialize runway geometry from four ordered corner points."""
        self.ps4 = ps4
        self.region_nums = region_nums
        self.col = col

        self.init_line_params()
        self.init_region_params()

    def init_region_params(self):
        """Precompute divided runway regions and inner region boundaries."""
        a, b = self.ps4[:2]
        self.len_AB = math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
        self.len_region = self.len_AB / self.region_nums

        x_ = (b[0] - a[0]) / self.region_nums
        y_ = (b[1] - a[1]) / self.region_nums
        ps_in_lineAB = [(a[0] + i * x_, a[1] + i * y_) for i in range(self.region_nums + 1)]
        ps_in_lineCD = [self._get_footpoint_in_line(p, self.line2_a, self.line2_b, self.line2_c)
                        for p in ps_in_lineAB]
        self.ps_in_lineAB = [(int(round(p[0])), int(round(p[1]))) for p in ps_in_lineAB]
        self.ps_in_lineCD = [(int(round(p[0])), int(round(p[1]))) for p in ps_in_lineCD]

        m = (1 - REGION_TH) / 2
        n = 1 - m

        dst_ps = [
            interpolate_pair_between_points(p1, p2, m, n)
            for p1, p2 in zip(self.ps_in_lineAB, self.ps_in_lineCD)
        ]
        self.ps_in_lineAB_closer = [tuple(map(round, p[0])) for p in dst_ps]
        self.ps_in_lineCD_closer = [tuple(map(round, p[1])) for p in dst_ps]

    def init_line_params(self):
        """Calculate the two runway edge line equations."""
        x1, y1 = self.ps4[0]
        x2, y2 = self.ps4[1]
        self.line_a = y2 - y1
        self.line_b = x1 - x2
        self.line_c = x2 * y1 - x1 * y2
        x1, y1 = self.ps4[2]
        x2, y2 = self.ps4[3]
        self.line2_a = y2 - y1
        self.line2_b = x1 - x2
        self.line2_c = x2 * y1 - x1 * y2

    def _get_footpoint_in_line(self, p, a, b, c):
        """Return the perpendicular projection of a point onto a line."""
        m, n = p
        denominator = a * a + b * b
        return ((b * b * m - a * b * n - a * c) / denominator,
                (a * a * n - a * b * m - b * c) / denominator)

    def draw_in_img(self, img, thickness=2, draw_bar=False, rect_4ps=None):
        """Draw runway boundaries and optional temperature bar/rectangle."""
        _2p = self.ps_in_lineAB[0], self.ps_in_lineAB[-1]
        cv2.line(img, *_2p, color=self.col, thickness=thickness)
        _2p = self.ps_in_lineCD[0], self.ps_in_lineCD[-1]
        cv2.line(img, *_2p, color=self.col, thickness=thickness)


        cv2.line(img, self.ps_in_lineAB[0], self.ps_in_lineCD[0], self.col, thickness)
        cv2.line(img, self.ps_in_lineAB[-1], self.ps_in_lineCD[-1], self.col, thickness)


        if rect_4ps is not None:
            cv2.line(img, rect_4ps[0], rect_4ps[1], color=self.col, thickness=thickness)
            cv2.line(img, rect_4ps[1], rect_4ps[2], color=self.col, thickness=thickness)
            cv2.line(img, rect_4ps[2], rect_4ps[3], color=self.col, thickness=thickness)
            cv2.line(img, rect_4ps[3], rect_4ps[0], color=self.col, thickness=thickness)

        if draw_bar:
            cv2.line(img, (BAR_ROI[2], BAR_ROI[0]), (BAR_ROI[2], BAR_ROI[1]), self.col, 1)
            cv2.line(img, (BAR_ROI[3], BAR_ROI[0]), (BAR_ROI[3], BAR_ROI[1]), self.col, 1)
            cv2.line(img, (BAR_ROI[2], BAR_ROI[0]), (BAR_ROI[3], BAR_ROI[0]), self.col, 1)
            cv2.line(img, (BAR_ROI[2], BAR_ROI[1]), (BAR_ROI[3], BAR_ROI[1]), self.col, 1)

        return img

    def get_rect_ps4(self):
        """Return the four corner points of the runway polygon."""
        return np.array([
            self.ps_in_lineAB[0],
            self.ps_in_lineAB[-1],
            self.ps_in_lineCD[-1],
            self.ps_in_lineCD[0],
        ])

class Video:
    """Shared video utilities for visible-light and infrared videos."""

    def _get_start_end_frame_id(self):
        """Set start and end frame indices from config time values."""
        start_time = parse_time_value(cfg['start_time'])
        end_time = parse_time_value(cfg['end_time'])
        self.start_frame = int(round(start_time * self.video_fps))
        self.end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_time:
            self.end_frame = int(round(end_time * self.video_fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def _get_wholemaskimg(self):
        """Build binary masks for all configured runways."""
        h, w = self.h, self.w
        masks = [
            cv2.fillConvexPoly(np.zeros([h, w], dtype=np.uint8), rw.get_rect_ps4(), 255)
            for rw in self.rws
        ]
        self.masks = [mask.astype(bool) for mask in masks]
        self.mask_all = np.zeros([h, w], dtype=np.uint8)
        for mask in self.masks:
            self.mask_all |= mask

    def init_bar_line(self, show_bar_line=False):
        """Extract the infrared temperature color bar from the current frame."""
        _, img = self.cap.read()
        if show_bar_line:
            temp = img.copy()
            cv2.line(temp, (BAR_ROI[2], 0), (BAR_ROI[2], self.h), (0, 0, 255), 1)
            cv2.line(temp, (BAR_ROI[3], 0), (BAR_ROI[3], self.h), (0, 0, 255), 1)
            cv2.line(temp, (BAR_ROI[2] - 10, BAR_ROI[0]), (BAR_ROI[3] + 10, BAR_ROI[0]), (0, 0, 255), 1)
            cv2.line(temp, (BAR_ROI[2] - 10, BAR_ROI[1]), (BAR_ROI[3] + 10, BAR_ROI[1]), (0, 0, 255), 1)
            cv2.imshow('Press Enter to continue...', temp)
            print('After confirming the temperature bar position, press any key to continue.')
            cv2.waitKey()
            cv2.destroyWindow('Press Enter to continue...')
        bar = img[BAR_ROI[0]:BAR_ROI[1], BAR_ROI[2]:BAR_ROI[3]]
        self.bar = np.mean(bar, 1).astype(np.float32)

    def comp_bg(self):
        """Load or compute the background image for the current video."""
        frames_num_used = 500

        if self.bg_img_path.exists():
            bg = cv2.imread(str(self.bg_img_path))
        else:
            print('Random pick up frames...')
            tim = time.time()
            inds = list(range(self.start_frame, self.end_frame))
            random.shuffle(inds)
            inds = inds[:frames_num_used]
            pbar = Pbar(total=len(inds))
            frames = []
            for i in inds:
                pbar.update()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame)
            pbar.close()
            print('Computing background image...')
            sx = stats.mode(np.array(frames))
            bg = sx[0][0]
            bg = cv2.medianBlur(bg, 3)
            cv2.imwrite(str(self.bg_img_path), bg)
            print(f'Background image calculation completed, time-consuming:{time.time() - tim}s')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.bg = bg.astype(np.int64)
        self.gray_bg_int16 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)

class VisibleLight(Video):
    """Visible-light mouse tracking paired with infrared temperature lookup."""

    def __init__(self, video_path):
        """Initialize visible-light tracking and matching infrared video."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_frames_num_src = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._get_start_end_frame_id()
        self.video_frames_num = self.end_frame - self.start_frame
        self.area_th = cfg['th_mouse_area']

        self.saved_dir = Path(Path(self.video_path).parent, Path(self.video_path).stem)
        self.saved_dir.mkdir(exist_ok=True)

        self.bg_img_path = Path(self.saved_dir, 'bg.bmp')
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        ps_config = cfg['ps']
        self.rws = build_runways()

        self._get_wholemaskimg()
        self.comp_bg()

        print('Load the infrared video information...')
        ir_video_path = cfg['video_path2']
        ir_bg_img_path = Path(self.saved_dir, 'bg_ir.bmp')
        self.ir = Infrared(ir_video_path, ir_bg_img_path)

        self.mouse_len = [[] for _ in ps_config]

    def run(self):
        """Track mouse positions frame by frame and save raw tracking output."""
        if SMALL_RECT_L is not None:
            self.get_small_rect_hw()

        pss_path = Path(self.saved_dir, TRACK_POINTS_FILE)
        self.track_pss = [[] for _ in self.rws]
        self.track_pss_smallrect_4ps = [[] for _ in self.rws]
        self.mouse_temp = [[] for _ in self.rws]

        bar = Pbar(total=self.video_frames_num)
        id_now = self.start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        capir = self.ir.cap
        capir.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        while True:
            has_frame, img = self.cap.read()
            if not has_frame:
                break
            has_ir_frame, fir = capir.read()
            if not has_ir_frame:
                break

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            last_mask = None
            rect_4ps = None

            for i, wms in enumerate(self.masks):
                mask = self._build_mouse_mask(img_gray, wms)
                last_mask = mask
                cp, avg_mouse_temp = self._track_mouse_in_runway(i, img, fir, mask)
                rect_4ps = self._record_frame_result(i, cp, avg_mouse_temp)

            for rw in self.rws:
                if SMALL_RECT_L is not None:
                    img = rw.draw_in_img(img, thickness=1, rect_4ps=rect_4ps)
                else:
                    img = rw.draw_in_img(img, thickness=1)
            if cfg['bin_img']:
                cv2.imshow('mask', last_mask)

            cv2.imshow('img', img)
            cv2.imshow('ir', fir)
            key = cv2.waitKeyEx(cfg['waittime'])

            if key == SPACE_KEY:
                cv2.waitKeyEx()

            bar.update()
            id_now += 1
            if id_now >= self.end_frame:
                break
        bar.close()

        self.track_pss = np.array(self.track_pss, np.float64)
        self.track_pss = np.round(self.track_pss, decimals=2)
        self.track_pss_smallrect_4ps = np.array(self.track_pss_smallrect_4ps)
        self.mouse_temp = np.array(self.mouse_temp, np.float64)
        self.mouse_temp = np.round(self.mouse_temp, decimals=2)

        np.save(pss_path, self.track_pss)

    def _build_mouse_mask(self, img_gray, runway_mask):
        """Build a binary mouse mask for one runway in a visible-light frame."""
        diff = np.abs(img_gray.astype(np.int16) - self.gray_bg_int16)
        mask = (diff > TH).astype(np.uint8) * 255
        mask = cv2.medianBlur(mask, 5)
        mask *= runway_mask

        kernel_size = cfg['erode_kernel_size']
        if kernel_size >= 3:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def _track_mouse_in_runway(self, runway_id, img, fir, mask):
        """Track one mouse in one runway and annotate the display frames."""
        retval, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if retval < 2:
            return OUTLIER, -1

        component_id = np.argmax(stats[1:, -1]) + 1
        area = stats[component_id, -1]
        x, y, w, h = stats[component_id, :4]

        roimask = mask[y:y + h, x:x + w]
        roiir = fir[y:y + h, x:x + w]
        max_n_coords, max_n_values = self.ir.get_max_n_pixel_temp_from_img(roiir, roimask)
        avg_mouse_temp = np.mean(max_n_values)

        hot_h = max_n_coords[0] + y
        hot_w = max_n_coords[1] + x
        img[hot_h, hot_w] = [0, 0, 255]
        fir[hot_h, hot_w] = [0, 0, 255]

        contours = self._get_contours(roimask, x, y)
        cv2.drawContours(img, contours, 0, (255, 0, 255), 1)
        cv2.drawContours(fir, contours, 0, (255, 255, 255), 1)

        self.mouse_len[runway_id].append(stats[component_id, 2])
        if area < self.area_th:
            return OUTLIER, avg_mouse_temp

        cp = centroids[component_id]
        cp_int = (int(cp[0]), int(cp[1]))
        cv2.circle(img, cp_int, 3, (255, 255, 255), -1)
        return cp, avg_mouse_temp

    @staticmethod
    def _get_contours(roimask, x_offset, y_offset):
        """Find contours in an ROI mask and shift them to full-frame coordinates."""
        contours_out = cv2.findContours(roimask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_out[1] if len(contours_out) == 3 else contours_out[0]
        for contour in contours:
            contour[:, :, 0] += x_offset
            contour[:, :, 1] += y_offset
        return contours

    def _record_frame_result(self, runway_id, cp, avg_mouse_temp):
        """Store per-frame position and temperature outputs for one runway."""
        self.track_pss[runway_id].append(cp)
        self.mouse_temp[runway_id].append(avg_mouse_temp)

        if SMALL_RECT_L is None:
            return None
        if cp is OUTLIER:
            rect_4ps = [OUTLIER for _ in range(4)]
        else:
            rect_4ps = self._get_4ps_from_cp(cp, self.rw_ks[runway_id])
        self.track_pss_smallrect_4ps[runway_id].append(rect_4ps)
        return [(int(round(p[0])), int(round(p[1]))) for p in rect_4ps]

    def get_small_rect_hw(self):
        """Calculate or load the small rectangle size used around each mouse."""
        if SMALL_RECT_L is None:
            mouselenavg = np.mean(np.array(self.mouse_len), axis=-1)
            scales = []
            for ps in cfg['ps']:
                a, b, c, d = cfg[ps]
                p1 = [(a[0] + d[0]) / 2, (a[1] + d[1]) / 2]
                p2 = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2]
                dif_y = abs(p1[1] - p2[1])
                dif_x = abs(p1[0] - p2[0])
                dif_z = (dif_x ** 2 + dif_y ** 2) ** 0.5
                scale = dif_z / dif_x
                scales.append(scale)
            self.small_rect_l = np.mean(mouselenavg * np.array(scales))
            print(f'small_rect_l: {self.small_rect_l}')
            raise SystemExit
        else:
            self.small_rect_l = SMALL_RECT_L
        print(f'---\nsmall_rect_l: {self.small_rect_l}')

        widths = []
        for rw in self.rws:
            a = rw.ps_in_lineAB[0]
            b = rw.ps_in_lineAB[-1]
            c = rw.ps_in_lineCD[0]
            d = rw.ps_in_lineCD[-1]
            l_ac = distance_between_points(a, c)
            l_bd = distance_between_points(b, d)
            widths.append((l_ac + l_bd) / 2 * cfg['region_th'])
        self.small_rect_width = np.array(widths).mean()
        print(f'small_rect_width: {self.small_rect_width}\n---')

        rw_ks = []
        for rw in self.rws:
            k1 = -(rw.line_a / rw.line_b)
            k2 = -(rw.line2_a / rw.line2_b)
            rw_ks.append((k1 + k2) / 2)
        self.rw_ks = rw_ks

    def _get_4ps_from_cp(self, cp, k):
        """Build a small rotated rectangle around the mouse center point."""
        l, w = self.small_rect_l, self.small_rect_width
        thet = math.atan(k)
        l2, w2 = l / 2, w / 2
        vk = [l2 * math.cos(thet), l2 * math.sin(thet)]
        if k >= 0:
            vkt = [w2 * math.sin(thet), -w2 * math.cos(thet)]
        else:
            vkt = [-w2 * math.sin(thet), w2 * math.cos(thet)]
        vk = np.array(vk)
        vkt = np.array(vkt)
        p = np.array(cp)
        p0 = p + vk + vkt
        p1 = p0 - 2 * vk
        p2 = p1 - 2 * vkt
        p3 = p2 + 2 * vk
        return p0, p1, p2, p3

    def _sec_to_minsec_str(self, sec):
        """Format seconds as MM:SS."""
        m, s = divmod(sec, 60)
        m = int(round(m))
        s = int(round(s))
        return f'{m:0>2d}:{s:0>2d}'

    def analyze(self):
        """Export per-frame mouse and local background temperatures."""
        cv2.destroyAllWindows()
        print(f'---\nAnalysis...')
        self.rect_temp = [[] for _ in self.rws]
        for i in range(len(self.rws)):
            for ps4 in self.track_pss_smallrect_4ps[i]:
                temp = self.ir.get_temp_from_rect_4ps(ps4)
                self.rect_temp[i].append(temp)

        duration_per_frame = 1 / self.video_fps
        frames_num_src = np.array(list(range(int(self.video_frames_num_src)))) + 1
        time_indexs_src = frames_num_src * duration_per_frame
        time_indexs = time_indexs_src[self.start_frame:self.end_frame]
        time_indexs = np.array([self._sec_to_minsec_str(x) for x in time_indexs])
        frame_indexs = np.array([list(range(self.start_frame, self.end_frame))])
        frame_indexs = np.squeeze(frame_indexs)

        for i, (rt, mt) in enumerate(zip(self.rect_temp, self.mouse_temp)):
            da = np.stack([frame_indexs, time_indexs, rt, mt], -1)
            df = pd.DataFrame(data=da, columns=['frame', 'time', 'backgroundtemp', 'mousetemp'])
            df.to_excel(Path(self.saved_dir, f'mousetemp_preframe_{i + 1}.xlsx'))
            print(f"The result has saved to {Path(self.saved_dir, f'mousetemp_preframe_{i + 1}.xlsx')}")
        print(f'Done\n---')


class Infrared(Video):
    """Infrared video processing and temperature conversion."""

    def __init__(self, video_path, bg_img_path):
        """Initialize infrared video resources and temperature calibration."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._get_start_end_frame_id()
        self.video_frames_num = self.end_frame - self.start_frame

        self.bg_img_path = Path(bg_img_path)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.saved_dir = Path(Path(self.video_path).parent, Path(self.video_path).stem)

        ps_config = cfg['ps']
        self.rws = build_runways()
        self._get_wholemaskimg()
        self.init_bar_line(show_bar_line=True)
        self.comp_bg()
        self.mask_all = self.mask_all.astype(bool)[:, :, None]

        self.lenbar = len(self.bar)
        self.temp_range = cfg['temp_range']
        self.range_dif = self.temp_range[1] - self.temp_range[0]
        self.temp_per_pixel = self.range_dif / self.lenbar
        self.pixel_max_n = cfg['pixel_max_n']

    def get_temp_from_rect_4ps(self, ps4):
        """Return average background temperature inside a four-point rectangle."""
        bg = self.bg
        maskall = self.mask_all
        ps4 = np.round(ps4).astype(np.int16)
        smallrect = cv2.fillConvexPoly(np.zeros(shape=maskall.shape, dtype=np.uint8), ps4, 255)
        smallrect = smallrect.astype(bool)
        dst = bg * maskall * smallrect
        pixel_num = (maskall * smallrect).sum()
        if pixel_num == 0:
            return cfg['outlier_temp']
        avg_value = np.sum(dst, axis=(0, 1)) / pixel_num
        avg_value = avg_value.reshape([1, 1, -1])
        avg_temp = self.pixelvalue_to_tempvalue(avg_value)[0, 0]
        return avg_temp

    def pixelvalue_to_tempvalue(self, ps):
        """Convert BGR pixel values to temperature values using the color bar."""
        sp_src = ps.shape
        ps = ps.reshape([-1, 3])
        dists = color_distance_to_palette(ps, self.bar)
        match_ids = np.argmin(dists, 1)
        match_ids = match_ids.reshape([*sp_src[:2]])
        temp = match_ids * self.temp_per_pixel + self.temp_range[0]
        temp = np.around(temp, decimals=2)
        return temp

    def get_max_n_pixel_temp_from_img(self, src, mask):
        """Return the hottest configured number of pixels inside a mask."""
        temps = self.pixelvalue_to_tempvalue(src)
        mask = mask.astype(bool)
        temps *= mask
        sp = temps.shape
        temps1d = temps.reshape([-1])
        max_n_ids = (-temps1d).argsort()[:self.pixel_max_n]
        max_n_coords = np.divmod(max_n_ids, sp[-1])
        max_n_values = temps[max_n_coords[0], max_n_coords[1]]
        return max_n_coords, max_n_values


def videos_have_same_frame_count(p1, p2):
    """Return whether two videos have the same frame count."""
    cap1 = cv2.VideoCapture(str(p1))
    cap2 = cv2.VideoCapture(str(p2))
    nb1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    nb2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
    cap1.release()
    cap2.release()
    return nb1 == nb2


def main():
    """Run the visible-light and infrared temperature analysis workflow."""
    if not videos_have_same_frame_count(cfg['video_path'], cfg['video_path2']):
        print('the frames number of two videos is not same!')
        print('exit.')
        raise SystemExit

    video = VisibleLight(cfg['video_path'])
    video.run()
    if SMALL_RECT_L is None:
        video.get_small_rect_hw()
    else:
        video.analyze()

if __name__ == '__main__':
    main()
