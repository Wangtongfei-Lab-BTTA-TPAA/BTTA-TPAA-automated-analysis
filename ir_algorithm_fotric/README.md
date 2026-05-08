# Infrared Fotric Analysis

Infrared video-analysis pipeline for extracting mouse feature-point
temperatures from Fotric-style thermal videos. The pipeline can split
non-runway videos, run DeepLabCut prediction, convert predicted coordinates into
temperature traces, and export CSV/figure results.

> This directory is intended for code and configuration. Large videos,
> generated outputs, and DeepLabCut model folders are local runtime files and
> should not be committed.

## Contents

- [Repository Structure](#repository-structure)
- [External Data And Model Folders](#external-data-and-model-folders)
- [Environment](#environment)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Input Layout](#input-layout)
- [Output Layout](#output-layout)
- [Related Papers](#related-papers)
- [GitHub Packaging Notes](#github-packaging-notes)

## Repository Structure

```text
ir_algorithm_fotric/
  1_processing.py       # Stage 1: split videos and run DeepLabCut prediction
  2_analysis.py         # Stage 2: extract and smooth temperature traces
  cfg.yaml              # Runtime configuration
  predict_videos.py     # Customized DeepLabCut prediction integration
  utils.py              # Shared helper functions
  README.md
```

`predict_videos.py` is based on DeepLabCut prediction code. Treat it as
vendor-style integration code and avoid editing it unless the prediction
workflow itself needs to change.

## External Data And Model Folders

During local use, the workflow expects two kinds of external folders:

```text
ir_algorithm_fotric/
  sample_data/          # local only: raw videos and generated outputs
  dlc_models/           # local only: trained DeepLabCut projects
```

Recommended local layout:

```text
ir_algorithm_fotric/
  sample_data/
    video_name.mp4
  dlc_models/
    your_dlc_project/
      config.yaml
```

## Environment

Run the scripts in a Python environment with the following packages available:

- DeepLabCut
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy
- scikit-image
- tqdm
- PyYAML

The DeepLabCut and TensorFlow versions should match the environment used to
train the selected model.

## Configuration

Edit `cfg.yaml` before running.

Most important fields:

| Field | Meaning |
| --- | --- |
| `Data_path` | Input video file or folder. |
| `Fmts` | Accepted video extensions, for example `.mp4` and `.avi`. |
| `Runway_video` | Whether the input is a runway video. |
| `Config_path` | Path to the trained DeepLabCut project `config.yaml`. |
| `Bar_roi` | Temperature bar ROI: `[y_start, y_end, x_start, x_end]`. |
| `Mask_polygon_vertexs` | Optional polygon mask for the valid detection area. |
| `Max_temp`, `Min_temp`, `Bar_mode` | Temperature-bar calibration settings. |
| `Split_point` | Horizontal split position for non-runway videos. |
| `R`, `N` | Local extraction radius and number of high-value pixels. |
| `Confidence_ths` | Confidence thresholds for filtering DLC points. |
| `Saved_img_quality` | JPEG quality for annotated preview frames. |

Example local paths:

```yaml
Data_path: D:\path\to\ir_algorithm_fotric\sample_data
Config_path: D:\path\to\ir_algorithm_fotric\dlc_models\your_dlc_project\config.yaml
```

Use an absolute path for `Config_path` when possible. Model training and project
organization should follow the official DeepLabCut workflow.

## Quick Start

Run the two stages in order:

```powershell
python 1_processing.py cfg.yaml
python 2_analysis.py cfg.yaml
```

Stage 1:

- Reads input videos from `Data_path`.
- For non-runway videos, creates left/right split videos.
- Runs DeepLabCut prediction.
- Saves DLC `.csv`, `.h5`, and metadata files.

Stage 2:

- Reads DLC CSV outputs.
- Saves corrected coordinates and annotated preview frames.
- Extracts raw temperature traces.
- Applies frequency-domain smoothing.
- Saves final CSV and figure outputs.

## Input Layout

If `Data_path` points to a folder, put raw videos directly inside that folder:

```text
sample_data/
  video_name.mp4
```

If `Data_path` points to a single video file, it can point directly to that
file.

For non-runway videos, stage 1 creates a generated result folder next to the raw
video:

```text
sample_data/
  video_name.mp4
  video_name/
    split_videos/
      video_name_1.mp4
      video_name_2.mp4
      split_line.jpg
```

`split_line.jpg` shows candidate split positions and the configured split line.

## Output Layout

For a non-runway source video named `video_name.mp4`, generated outputs follow
this structure:

```text
sample_data/
  video_name.mp4
  video_name/
    split_videos/
      video_name_1.mp4
      video_name_1DLC_*.csv
      video_name_1DLC_*.h5
      video_name_1DLC_*_meta.pickle
      video_name_2.mp4
      video_name_2DLC_*.csv
      video_name_2DLC_*.h5
      video_name_2DLC_*_meta.pickle
      split_line.jpg
    frames/
      0.jpg
      1.jpg
      ...
    coordinates.csv
    draw_bar.tif
    fft_1.png
    fft_2.png
    smooth_result_1.png
    smooth_result_2.png
    temperature.csv
    temperature_smooth.csv
```

Typical output files:

| File | Description |
| --- | --- |
| `coordinates.csv` | Corrected feature-point coordinates, confidence values, and filtration flags. |
| `temperature.csv` | Raw extracted temperature traces and filtration flags. |
| `temperature_smooth.csv` | Frequency-filtered temperature traces and filtration flags. |
| `fft_*.png` | Frequency spectra used during smoothing. |
| `smooth_result_*.png` | Raw and smoothed trace visualization. |
| `draw_bar.tif` | First frame with the configured temperature bar ROI highlighted. |
| `frames/*.jpg` | Annotated coordinate preview frames. |
| `split_videos/*DLC*.csv` | DeepLabCut coordinate output used by stage 2. |

## Related Papers

- DeepLabCut method paper: [DeepLabCut: markerless pose estimation of user-defined body parts with deep learning](https://doi.org/10.1038/s41593-018-0209-y).
- DeepLabCut protocol paper: [Using DeepLabCut for 3D markerless pose estimation across species and behaviors](https://doi.org/10.1038/s41596-019-0176-0).

## GitHub Packaging Notes

Do not commit local data, trained models, generated outputs, or Python cache
files:

```text
sample_data/
dlc_models/
__pycache__/
*.pyc
```

Users should prepare their own videos and DeepLabCut model folder, then update
`Data_path` and `Config_path` in `cfg.yaml`.
