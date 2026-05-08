# Video Analysis Methods

This repository contains two Python video-analysis workflows used for mouse
thermal behavior and infrared temperature extraction.

## Algorithms

| Directory | Purpose | Main entry |
| --- | --- | --- |
| `ir_algorithm_fotric/` | Extract feature-point temperatures from Fotric-style infrared videos with DeepLabCut-assisted tracking. | `python 1_processing.py cfg.yaml`, then `python 2_analysis.py cfg.yaml` |
| `temperature_preference_code/` | Analyze paired visible-light and infrared videos from a mouse temperature-preference runway assay. | `python main.py` |

Each directory has its own README with configuration details, expected input
layout, output files, and runtime notes.

## Repository Layout

```text
CIBR_WangtongfeiLab_method/
  README.md
  .gitignore
  ir_algorithm_fotric/
    README.md
    cfg.yaml
    1_processing.py
    2_analysis.py
  temperature_preference_code/
    README.md
    config.yaml
    main.py
    bg_repair.py
```

## Environment

The workflows are regular Python scripts. Create a project-specific Python
environment and install the dependencies listed in each algorithm README.

The infrared Fotric workflow additionally depends on a trained DeepLabCut
project. Keep trained models and raw videos outside version control unless a
small public example is intentionally included.

## Quick Start

Infrared Fotric workflow:

```powershell
cd ir_algorithm_fotric
python 1_processing.py cfg.yaml
python 2_analysis.py cfg.yaml
```

Temperature-preference workflow:

```powershell
cd temperature_preference_code
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
