# Temperature Preference Analysis

Video-analysis pipeline for paired visible-light and infrared recordings from a
mouse temperature-preference runway assay. The visible-light video is used for
mouse segmentation and tracking; the infrared video is used to estimate mouse
body temperature and local background temperature.

> This directory is intended for code, configuration, and small examples. Raw
> experiment videos, generated outputs, and local cache files can become large
> and should usually stay outside version control.

## Contents

- [Repository Structure](#repository-structure)
- [Environment](#environment)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Input Layout](#input-layout)
- [Output Layout](#output-layout)
- [Related Papers](#related-papers)
- [GitHub Packaging Notes](#github-packaging-notes)

## Repository Structure

```text
temperature_preference_code/
  main.py          # Main paired visible-light/infrared analysis workflow
  bg_repair.py     # Interactive helper for repairing generated backgrounds
  config.yaml      # Runtime configuration
  utils.py         # Shared helper functions
  README.md
```

## Environment

Run the scripts in a Python environment with the following packages available:

- OpenCV (`opencv-python`)
- NumPy
- Pandas
- PyYAML
- SciPy
- Excel writer support for `pandas.to_excel`, such as `openpyxl`

OpenCV preview windows are used during analysis, so a desktop session is
recommended for normal operation.

## Configuration

Edit `config.yaml` before running.

Most important fields:

| Field | Meaning |
| --- | --- |
| `video_path` | Visible-light video path. |
| `video_path2` | Infrared video path. |
| `start_time`, `end_time` | Time range to process. Use `MM:SS`, `HH:MM:SS`, or `None` for `end_time`. |
| `ps1`, `ps2`, `ps` | Runway corner coordinates and selected runways. |
| `bar_roi` | Temperature color-bar ROI: `[y_start, y_end, x_start, x_end]`. |
| `temp_range` | Temperature range represented by the infrared color bar. |
| `small_rect_l` | Local background rectangle length. Set to `None` to estimate it. |
| `th`, `th_mouse_area`, `erode_kernel_size` | Visible-light mouse segmentation controls. |

Paths containing Chinese characters are not supported by the current workflow.

## Quick Start

Run the main workflow from this directory:

```powershell
python main.py
```

The script opens preview windows for the visible-light video, infrared video,
and optionally the binary segmentation mask. Press the space key to pause
playback.

If a generated background image contains artifacts, use the interactive repair
helper:

```powershell
python bg_repair.py
```

## Input Layout

The workflow expects one visible-light video and one infrared video with the
same frame count:

```text
sample_data/
  experiment-Light.avi
  experiment-IR.avi
```

Set `video_path` and `video_path2` in `config.yaml` to the corresponding files.

## Output Layout

Generated outputs are saved beside the visible-light video in a folder named
after the visible-light video stem:

```text
sample_data/
  experiment-Light.avi
  experiment-IR.avi
  experiment-Light/
    bg.bmp
    bg_ir.bmp
    bg_old.bmp
    bg_ir_old.bmp
    track_pss.npy
    mousetemp_preframe_1.xlsx
    mousetemp_preframe_2.xlsx
```

Typical output files:

| File | Description |
| --- | --- |
| `bg.bmp` | Computed visible-light background image. |
| `bg_ir.bmp` | Computed infrared background image. |
| `track_pss.npy` | Tracked mouse center points. |
| `mousetemp_preframe_*.xlsx` | Per-frame local background and mouse temperature table. |
| `bg_old.bmp`, `bg_ir_old.bmp` | Background backups created by `bg_repair.py`. |

## GitHub Packaging Notes

Do not commit local experiment data, generated outputs, or Python cache files:

```text
sample_data/*/
__pycache__/
*.pyc
```

Keep `config.yaml` as an editable example. Users should update video paths,
runway coordinates, and color-bar settings for their own experiments.
