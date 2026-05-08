#! python
"""
Module Name: 2_analysis
Affiliation: CIBR_WangtongfeiLab

Description:
    Convert DeepLabCut coordinate output into corrected coordinate previews,
    extract infrared temperature traces around feature points, apply frequency
    filtering, and export CSV and figure results for each video.

Functions:
    - init_config, load_result_from_csv, get_dlc_results: Configuration and DLC IO.
    - refine_point, get_centroid_of_img: Coordinate refinement helpers.
    - PixelValueToTempValue: Temperature-bar calibration and pixel conversion.
    - get_temperature, save_temperature_csvs, save_smooth_plots: Temperature export.
    - smooth, find_zero_interval: Signal filtering and plotting helpers.

Usage:
    python 2_analysis.py cfg.yaml

Copyright:
    Copyright (c) CIBR_WangtongfeiLab.

Note:
    If this code is used in academic publication,
    please cite or acknowledge CIBR_WangtongfeiLab.
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from sklearn.linear_model import LinearRegression

try:
    from utils import load_cfg, get_video_paths, Pbar
except ImportError:
    import utils

    load_cfg = utils.load_cfg
    get_video_paths = utils.get_video_paths
    Pbar = utils.Pbar


cfg = None
Feature_point_num = None
Colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 255),
    (255, 0, 0),
]


def init_config(config_path):
    """Load configuration and initialize derived script-level settings."""
    global cfg, Feature_point_num

    cfg = load_cfg(config_path)

    is_linux = cfg.Config_path[0] in ['/', '\\']
    if is_linux:
        cfg.Show = False

    if cfg.Runway_video:
        Feature_point_num = 3
    else:
        Feature_point_num = 2


def load_result_from_csv(csv_path, return_confidence=False, feature_points_num=None):
    """Read a DeepLabCut CSV file and return point coordinates and confidences."""
    if feature_points_num is None:
        feature_points_num = Feature_point_num

    with open(str(csv_path), 'r', encoding='utf-8') as f:
        lines = f.readlines()[3:]

    lines = np.array([line.split(',') for line in lines])
    confidences = [lines[:, 3 * i].astype(float) for i in range(1, feature_points_num + 1)]
    points = [lines[:, i * 3 + 1:i * 3 + 3].astype(float) for i in range(feature_points_num)]
    points = np.round(np.concatenate(points, -1)).astype(int)
    confidences = np.stack(confidences, -1)

    if return_confidence:
        return points, confidences
    return points


def get_dlc_results(video_path):
    """Collect DLC coordinate and confidence results for one source video."""
    video_path = Path(video_path)

    if not cfg.Runway_video:
        result_paths = sorted((video_path.parent / video_path.stem / 'split_videos').glob('*.csv'))
        result_paths = [path for path in result_paths if 'DLC' in path.name]
        results = [load_result_from_csv(path, return_confidence=True) for path in result_paths]
        confidences = np.array([result[1] for result in results])
        points = np.array([result[0] for result in results])
    else:
        result_path = sorted(video_path.parent.glob(f'{video_path.stem}*.csv'))[0]
        points, confidences = load_result_from_csv(result_path, return_confidence=True)
        confidences = confidences[None]
        points = points[None]

    return confidences, points


def get_DLCres(video_path):
    """Backward-compatible wrapper for older external callers."""
    return get_dlc_results(video_path)


def get_time_point_min(frame_id, fps):
    """Convert a frame index into elapsed time in minutes."""
    return round(frame_id / fps / 60, 4)


def get_centroid_of_img(frame):
    """Calculate a positive-intensity centroid from an image patch."""
    matrix = frame.astype(float)
    matrix -= matrix.mean()
    matrix = np.clip(matrix, 0, None)
    if len(matrix.shape) == 3:
        matrix = matrix.sum(-1)

    row = matrix.sum(0)
    col = matrix.sum(1)
    if row.sum() == 0 or col.sum() == 0:
        return False, 0, 0

    x = (row * np.arange(len(row))).sum() / row.sum()
    y = (col * np.arange(len(col))).sum() / col.sum()
    return True, int(round(x)), int(round(y))


def refine_point(frame, point, radius=None):
    """Refine a detected point by searching the local circular neighborhood."""
    if radius is None:
        radius = cfg.Refine_R

    h, w = frame.shape[:2]
    h0 = point[1] - radius if point[1] - radius > 0 else 0
    w0 = point[0] - radius if point[0] - radius > 0 else 0
    h1 = point[1] + radius if point[1] + radius < h else h
    w1 = point[0] + radius if point[0] + radius < w else w

    roi = frame[h0:h1, w0:w1].copy()
    mask = np.zeros([roi.shape[0], roi.shape[1]], np.uint8)
    cv2.circle(mask, (point[0] - w0, point[1] - h0), radius, 255, -1)
    roi *= mask.astype(bool)[..., None]

    ok, x, y = get_centroid_of_img(roi)
    if ok:
        return x + w0, y + h0
    return point


def show_result():
    """Save corrected coordinates and annotated preview frames for each video."""
    videos = get_video_paths(cfg)
    for video_path in videos:
        print(f'Generating coordinate preview: {video_path}')
        confidences, results = get_dlc_results(video_path)

        video_path = Path(video_path)
        result_dir = video_path.parent / video_path.stem
        result_dir.mkdir(exist_ok=True)
        frames_dir = result_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        csv_file = result_dir / 'coordinates.csv'

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if not cfg.Runway_video:
            line_x = int(width * cfg.Split_point)
            results[1, :, 0] += line_x
            results[1, :, 2] += line_x

        frame_count = results.shape[1]
        print(f'Total frames: {frame_count}')

        with open(str(csv_file), 'w', encoding='utf-8', newline='') as csv:
            if cfg.Runway_video:
                csv.write(f'Frame,Time-min{",x,y,Confidence,Filtration" * Feature_point_num}\n')
            else:
                csv.write(f'Frame,Time-min{",x,y,Confidence,Filtration" * Feature_point_num * 2}\n')

            pbar = Pbar(frame_count)
            for frame_id in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                csv.write(f'{frame_id + 1},{get_time_point_min(frame_id, fps)}')
                for result, confidence in zip(results, confidences):
                    for point_id in range(Feature_point_num):
                        point = (result[frame_id][point_id * 2], result[frame_id][point_id * 2 + 1])
                        new_point = point if cfg.Runway_video else refine_point(frame, point)
                        confidence_value = confidence[frame_id][point_id]
                        filtration = 1 if confidence_value < cfg.Confidence_ths[point_id] else 0

                        csv.write(f',{new_point[0]},{new_point[1]},{confidence_value:.3f},{filtration}')
                        cv2.putText(frame, f'{confidence_value:.3f}', point, 1, 1, (255, 255, 255), lineType=cv2.LINE_AA)
                        cv2.circle(frame, point, int(round(50 * confidence_value)), Colors[point_id], 1, lineType=cv2.LINE_AA)
                        cv2.circle(frame, new_point, 2, Colors[point_id], -1, lineType=cv2.LINE_AA)

                csv.write('\n')
                cv2.imwrite(str(frames_dir / f'{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.Saved_img_quality])
                pbar.update()

            pbar.close()

        cap.release()
        print()


class PixelValueToTempValue:
    """Convert grayscale or color-bar pixel values into temperature values."""

    def __init__(self, frame):
        """Initialize the conversion model from the configured temperature bar."""
        frame = frame.copy()
        bar_roi = cfg.Bar_roi

        if cfg.Runway_video:
            self.bar = frame[bar_roi[0]:bar_roi[1], bar_roi[2]:bar_roi[3]]
            self.bar = np.mean(self.bar.astype(float), 1)
            self.temp_per_pixel = (cfg.Max_temp - cfg.Min_temp) / len(self.bar)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.bar = frame[bar_roi[0]:bar_roi[1], bar_roi[2]:bar_roi[3]]
            self.init_gray_bar_line(self.bar)

    def init_gray_bar_line(self, bar):
        """Fit a linear grayscale-to-temperature model from the bar ROI."""
        bar = bar.astype(float).mean(1)
        if cfg.Bar_mode == 0:
            temps = np.linspace(cfg.Min_temp, cfg.Max_temp, len(bar))
        else:
            temps = np.linspace(cfg.Max_temp, cfg.Min_temp, len(bar))

        model = LinearRegression()
        model.fit(bar.reshape([-1, 1]), temps.reshape([-1, 1]))
        self.bar_line_intercept = model.intercept_[0]
        self.bar_line_slope = model.coef_[0, 0]

    def gray_call(self, pixel_values):
        """Convert grayscale pixel values to temperatures."""
        pixel_values = np.array(pixel_values, float)
        return pixel_values * self.bar_line_slope + self.bar_line_intercept

    def color_call(self, pixel_values):
        """Convert BGR pixel values to temperatures by matching the color bar."""
        source_shape = pixel_values.shape
        assert source_shape[-1] == 3, 'pixel_values.shape[-1] != 3'

        pixels = pixel_values.reshape([-1, 3])
        distances = color_distance(pixels, self.bar)
        match_ids = np.argmin(distances, 1).reshape([*source_shape[:2]])

        if cfg.Bar_mode == 0:
            return match_ids * self.temp_per_pixel + cfg.Min_temp
        return cfg.Max_temp - match_ids * self.temp_per_pixel

    def __call__(self, pixel_values):
        """Dispatch to the grayscale or color conversion path."""
        if cfg.Runway_video:
            return self.color_call(pixel_values)
        return self.gray_call(pixel_values)


Pixelvalue_to_tempvalue = PixelValueToTempValue


def bgr_distance(point, points):
    """Calculate weighted BGR distance from one color to a set of colors."""
    b1, g1, r1 = point
    b2, g2, r2 = points[:, 0], points[:, 1], points[:, 2]
    rm = (r1 + r2) / 2
    b = b1 - b2
    g = g1 - g2
    r = r1 - r2
    return (2 + rm / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rm) / 256) * (b ** 2)


def color_distance(bgrs, color_bar):
    """Calculate distances between image colors and all colors on the bar."""
    return np.stack([bgr_distance(color, bgrs) for color in color_bar], axis=-1)


def get_mask():
    """Create the circular mask used for local point temperature extraction."""
    mask = np.zeros([2 * cfg.R + 1, 2 * cfg.R + 1], np.uint8)
    cv2.circle(mask, (cfg.R, cfg.R), cfg.R, 255, -1)
    if cfg.Runway_video:
        return mask.astype(bool)[..., None]
    return mask.astype(bool)


def get_rois_from_points(img, points, hw=140):
    """Crop fixed-size ROIs centered around feature points."""
    h, w = img.shape[:2]
    points = np.array(points).reshape([-1, 2])
    rois = []

    for point in points:
        x, y = point
        radius = int(hw / 2)
        h0, h1 = y - radius, y - radius + hw
        w0, w1 = x - radius, x - radius + hw

        if h0 < 0:
            h0 = 0
            h1 = hw
        elif h1 > h:
            h1 = h
            h0 = h - hw

        if w0 < 0:
            w0 = 0
            w1 = hw
        elif w1 > w:
            w1 = w
            w0 = w - hw

        rois.append([img[h0:h1, w0:w1], [h0, h1, w0, w1]])

    return rois


def draw_bar_roi(frame, video_path):
    """Save a frame with the configured temperature bar ROI highlighted."""
    bar_roi = cfg.Bar_roi
    frame[bar_roi[0], bar_roi[2]:bar_roi[3]] = np.array([0, 0, 255], dtype=frame.dtype)
    frame[bar_roi[1], bar_roi[2]:bar_roi[3]] = np.array([0, 0, 255], dtype=frame.dtype)
    frame[bar_roi[0]:bar_roi[1], bar_roi[2]] = np.array([0, 0, 255], dtype=frame.dtype)
    frame[bar_roi[0]:bar_roi[1], bar_roi[3]] = np.array([0, 0, 255], dtype=frame.dtype)
    cv2.imwrite(str(video_path.parent / video_path.stem / 'draw_bar.tif'), frame)


def get_temperature():
    """Extract raw and smoothed temperature traces for all configured videos."""
    videos = get_video_paths(cfg)
    for video_path in videos:
        print(f'Calculating temperature: {video_path}')
        video_path = Path(video_path)
        stem = video_path.stem

        _, results = get_dlc_results(video_path)

        coord_path = video_path.parent / stem / 'coordinates.csv'
        coord = pd.read_csv(str(coord_path))
        filtration = coord[[col for col in coord.columns if col.startswith('Filtration')]].values

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = results.shape[1]

        if not cfg.Runway_video:
            line_x = int(width * cfg.Split_point)
            results[1, :, 0] += line_x
            results[1, :, 2] += line_x

        ret, first_frame = cap.read()
        pixel_to_temp = PixelValueToTempValue(first_frame)
        draw_bar_roi(first_frame, video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        pixel_values = []
        mask = get_mask()
        pbar = Pbar(frame_count)
        for frame_id in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            if not cfg.Runway_video:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mouse_rois = [get_rois_from_points(frame, points, hw=2 * cfg.R + 1) for points in results[:, frame_id]]
            frame_values = []
            for rois in mouse_rois:
                mouse_values = []
                for point_id, (roi_src, roi_hw) in enumerate(rois):
                    roi = roi_src.copy()
                    h, w = roi.shape[:2]

                    if cfg.Runway_video:
                        roi = pixel_to_temp(roi)
                        if point_id in cfg.Tail_id:
                            mouse_values.append(roi[cfg.R, cfg.R])
                        else:
                            roi_src = np.round(roi * 255 / cfg.Max_temp).astype(np.uint8)
                            roi *= mask[..., 0]
                            roi = np.reshape(roi, [-1])
                            inds = np.argpartition(roi, -cfg.N)[-cfg.N:]
                            values = [roi[ind] for ind in inds]
                            mouse_values.append(np.mean(values))
                    else:
                        roi *= mask
                        roi = np.reshape(roi, [-1])
                        inds = np.argpartition(roi, -cfg.N)[-cfg.N:]
                        values = [roi[ind] for ind in inds]
                        mouse_values.append(np.mean(values))

                    if cfg.Show and not (cfg.Runway_video and point_id in cfg.Tail_id):
                        roi_src_rsp = np.reshape(roi_src, [-1])
                        roi_src_rsp[inds] = 255
                        v255 = np.reshape(roi_src_rsp.copy(), (h, w))
                        roi_src_rsp[inds] = 0
                        v0 = np.reshape(roi_src_rsp.copy(), (h, w))
                        display_frame[roi_hw[0]:roi_hw[1], roi_hw[2]:roi_hw[3], 0] = v255
                        display_frame[roi_hw[0]:roi_hw[1], roi_hw[2]:roi_hw[3], 1] = v255
                        display_frame[roi_hw[0]:roi_hw[1], roi_hw[2]:roi_hw[3], 2] = v0

                frame_values.append(mouse_values)
            pixel_values.append(frame_values)

            if cfg.Show:
                for points in results[:, frame_id]:
                    points = np.array(points).reshape([-1, 2])
                    for point_id, point in enumerate(points):
                        cv2.circle(display_frame, (point[0], point[1]), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                        if not (cfg.Runway_video and point_id in cfg.Tail_id):
                            cv2.circle(display_frame, (point[0], point[1]), cfg.R, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                cv2.imshow('frame', display_frame)
                cv2.waitKey(cfg.Show_waittime)

            pbar.update()
        pbar.close()
        cap.release()

        if cfg.Runway_video:
            temps = np.array(pixel_values)
        else:
            temps = pixel_to_temp(pixel_values)

        temps = np.reshape(temps, [temps.shape[0], -1])
        smooth_results = []
        filter_ids = [[] for _ in range(temps.shape[1])]
        for point_id in range(temps.shape[1]):
            filtration_mask = filtration[:, point_id]
            values = temps[:, point_id]
            filter_values = []
            for value_id in range(len(values)):
                if not filtration_mask[value_id]:
                    filter_values.append(values[value_id])
                    filter_ids[point_id].append(value_id)
            smooth_results.append(smooth(filter_values, str(video_path.parent / stem / f'fft_{point_id + 1}.png'), fs=fps))

        save_smooth_plots(video_path, temps, smooth_results, filter_ids, filtration)
        save_temperature_csvs(video_path, temps, smooth_results, filter_ids, filtration, fps)


def save_smooth_plots(video_path, temps, smooth_results, filter_ids, filtration):
    """Save per-point plots comparing raw and filtered temperature traces."""
    for point_id in range(temps.shape[1]):
        src = temps[:, point_id]
        dst = smooth_results[point_id]
        dst_ids = filter_ids[point_id]
        filt = filtration[:, point_id]

        plt.close()
        plt.figure(figsize=[15, 8])
        plt.plot(src)
        for idx in range(1, len(dst_ids)):
            if dst_ids[idx] - dst_ids[idx - 1] == 1:
                plt.plot([dst_ids[idx - 1], dst_ids[idx]], [dst[idx - 1], dst[idx]], c='r')

        min_y, max_y = src.min(), src.max()
        for start, end in find_zero_interval(filt):
            if start == end:
                min_x, max_x = start - 0.5, start + 0.5
            else:
                min_x, max_x = start, end
            plt.fill([min_x, max_x, max_x, min_x], [min_y, min_y, max_y, max_y], 'black', alpha=0.3)

        plt.savefig(video_path.parent / video_path.stem / f'smooth_result_{point_id + 1}.png')


def save_temperature_csvs(video_path, temps, smooth_results, filter_ids, filtration, fps):
    """Save raw and smoothed temperature CSV files for one video."""
    stem = video_path.stem
    frame_id = np.arange(1, temps.shape[0] + 1)[..., None]
    time_min = np.array([get_time_point_min(i, fps) for i in range(temps.shape[0])])[..., None]

    temps_and_filtration = np.zeros([temps.shape[0], temps.shape[1] * 2], temps.dtype)
    columns = ['Frame', 'Time-min']
    for point_id in range(temps.shape[1]):
        temps_and_filtration[:, point_id * 2] = temps[:, point_id]
        temps_and_filtration[:, point_id * 2 + 1] = filtration[:, point_id]
        columns += [str(point_id + 1), 'Filtration']

    df = pd.DataFrame(data=np.concatenate([frame_id, time_min, temps_and_filtration], -1), columns=columns)
    df.to_csv(str(video_path.parent / stem / 'temperature.csv'), index=False)

    result = np.ones_like(temps) * np.nan
    for point_id in range(temps.shape[1]):
        data = iter(smooth_results[point_id])
        for frame_idx in filter_ids[point_id]:
            result[frame_idx, point_id] = next(data)

    result_and_filtration = np.zeros([result.shape[0], result.shape[1] * 2], result.dtype)
    columns = ['Frame', 'Time-min']
    for point_id in range(result.shape[1]):
        result_and_filtration[:, point_id * 2] = result[:, point_id]
        result_and_filtration[:, point_id * 2 + 1] = filtration[:, point_id]
        columns += [str(point_id + 1), 'Filtration']

    df = pd.DataFrame(data=np.concatenate([frame_id, time_min, result_and_filtration], -1), columns=columns)
    df.to_csv(str(video_path.parent / stem / 'temperature_smooth.csv'), index=False)


def find_zero_interval(values):
    """Return continuous zero-valued intervals from a one-dimensional mask."""
    img = np.array(values, np.uint8)[None]
    _, _, stats, _ = cv2.connectedComponentsWithStats(img)
    if len(stats) <= 1:
        return []
    return [[stat[0], stat[0] + stat[2] - 1] for stat in stats[1:]]


def smooth(xs, fft_img_save_path, fs, ranges=None):
    """Filter a temperature signal in the frequency domain and save its spectrum."""
    if ranges is None:
        ranges = cfg.Ranges

    if len(xs) <= 5:
        return xs

    valid_ranges = []
    range_max = min(100, int(len(xs) / 2))
    for start, end in ranges:
        if start >= range_max:
            continue
        if end > range_max:
            valid_ranges.append([start, range_max])
        else:
            valid_ranges.append([start, end])
    ranges = valid_ranges

    fft_data = np.fft.fft(xs)
    shifted_fft = np.fft.fftshift(fft_data)
    amplitudes = np.abs(shifted_fft)
    center_id = int(len(amplitudes) / 2)
    max_amplitude = amplitudes[:center_id].max() * 1.2

    freq_fn = lambda index: index * fs / len(xs)
    if len(xs) % 2 == 0:
        x_freq = [freq_fn(i) for i in range(center_id)]
        x_freq = list(reversed([-x for x in x_freq])) + x_freq
    else:
        x_freq = [freq_fn(i) for i in range(center_id + 1)]
        x_freq = list(reversed([-x for x in x_freq][1:])) + x_freq

    x_location = list(range(-center_id, center_id + 1))[:len(amplitudes)]

    plt.close()
    plt.figure(figsize=[15, 8])
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    if len(amplitudes) > 200:
        plot_start = int((len(amplitudes) - 200) / 2)
        x_location = x_location[plot_start:plot_start + 200]
        y = amplitudes[plot_start:plot_start + 200]
        x_freq = x_freq[plot_start:plot_start + 200]
    else:
        y = amplitudes

    plt.plot(x_location, y)
    plt.xlabel('Location')
    plt.ylabel('Amplitude')
    plt.ylim((0, max_amplitude))
    for low, high in ranges:
        plt.fill([low, high, high, low], [0, 0, max_amplitude, max_amplitude], 'red', alpha=0.2)
        plt.fill([-low, -high, -high, -low], [0, 0, max_amplitude, max_amplitude], 'red', alpha=0.2)

    ax2 = plt.twiny()
    ax2.plot(x_freq, y, alpha=0)
    ax2.set_xlabel('Hz')
    plt.savefig(str(fft_img_save_path))

    mask = np.zeros_like(shifted_fft).astype(bool)
    center = int(len(mask) / 2)
    for low, high in ranges:
        mask[center + low:center + high] = True
        mask[center - high:center - low] = True

    shifted_fft = shifted_fft * mask
    inverse_shift = np.fft.ifftshift(shifted_fft)
    dst = np.fft.ifft(inverse_shift)
    return np.abs(dst)


def main():
    """Load configuration and run coordinate preview plus temperature extraction."""
    if len(sys.argv) < 2:
        print('Usage: python 2_analysis.py <cfg.yaml>')
        return

    init_config(sys.argv[1])
    start_time = time.time()

    show_result()
    get_temperature()

    end_time = time.time()
    print(f'Elapsed time: {(end_time - start_time) / 60:.2f} mins')


if __name__ == '__main__':
    main()
