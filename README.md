# BTTA-TPAA-automated-analysis

This repository contains two Python video-analysis workflows used for mouse
thermal behavior and infrared temperature extraction.

## Algorithms

| Directory | Purpose | Main entry |
| --- | --- | --- |
| `BAT-Tail Temperature Analysis Algorithm/` | Extract feature-point temperatures from Fotric-style infrared videos with DeepLabCut-assisted tracking. | `python 1_processing.py cfg.yaml`, then `python 2_analysis.py cfg.yaml` |
| `Temperature Preference Analysis Algorithm/` | Analyze paired visible-light and infrared videos from a mouse temperature-preference runway assay. | `python main.py` |

Each directory has its own README with configuration details, expected input
layout, output files, and runtime notes.

## Repository Layout

```text
CIBR_WangtongfeiLab_method/
  README.md
  .gitignore
  BAT-Tail Temperature Analysis Algorithm/
    README.md
    cfg.yaml
    1_processing.py
    2_analysis.py
  Temperature Preference Analysis Algorithm/
    README.md
    config.yaml
    main.py
    bg_repair.py
```

## Environment

The workflows are regular Python scripts. Create a project-specific Python
environment and install the dependencies listed in each algorithm README.

The BAT-Tail Temperature Analysis Algorithm additionally depends on a trained
DeepLabCut project. Keep trained models and raw videos outside version control
unless a small public example is intentionally included.

## Quick Start

BAT-Tail Temperature Analysis Algorithm:

```powershell
cd "BAT-Tail Temperature Analysis Algorithm"
python 1_processing.py cfg.yaml
python 2_analysis.py cfg.yaml
```

Temperature Preference Analysis Algorithm:

```powershell
cd "Temperature Preference Analysis Algorithm"
python main.py
```

Before running, update the YAML configuration file in the selected directory to
match local video paths, temperature color-bar settings, and experiment
geometry.

## Open-source Notes

- Keep README files, configuration examples, and code comments in English for
  easier GitHub reuse.
- Do not commit raw experiment videos, generated result folders, trained models,
  or cache files.
- If this code is used in an academic publication, cite the relevant method and
  assay papers above, and acknowledge CIBR_WangtongfeiLab as appropriate.
