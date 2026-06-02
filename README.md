# AI-Based Dental Plaque Level Determination

AI-assisted dental plaque analysis system for visually impaired users, developed during an internship at **PSU-Dolphins Co., Ltd.**, in collaboration with the **Faculty of Dentistry, Prince of Songkla University** and the **Faculty of Dentistry, Mahidol University**.

This project extends Mask R-CNN tooth segmentation with automated plaque detection, PHP/QHPI scoring, clinical dashboard visualization, and brushing recommendation generation.

> This project is intended for research and educational use. Results should be reviewed and validated by qualified dental professionals before clinical application.

---

## Overview

The system analyzes dental photographs after plaque-disclosing treatment and estimates plaque accumulation on the anterior upper teeth.

The current workflow supports:

* Tooth segmentation using Mask R-CNN
* Automatic identification of upper anterior teeth using FDI tooth numbering
* Plaque detection from disclosed plaque colors
* PHP (Patient Hygiene Performance) scoring
* QHPI (Quigley-Hein Plaque Index) scoring
* Tooth-level visualization
* Case summary dashboard generation
* Brushing recommendation generation

---

## Target Teeth

The current implementation focuses on six upper anterior teeth:

| FDI Code | Tooth                       |
| -------- | --------------------------- |
| 13       | Upper right canine          |
| 12       | Upper right lateral incisor |
| 11       | Upper right central incisor |
| 21       | Upper left central incisor  |
| 22       | Upper left lateral incisor  |
| 23       | Upper left canine           |

---

## Pipeline Workflow

```text
Dental Photograph
        ↓
Mask R-CNN Tooth Segmentation
        ↓
Smart FDI Tooth Assignment
        ↓
Plaque Detection
        ↓
PHP and QHPI Scoring
        ↓
Visualization and Summary Dashboard
        ↓
Brushing Recommendation Generation
```

The automated main pipeline contains three stages:

```text
Step 1: Run_Model_1Class.py
        → Tooth segmentation and tooth image extraction

Step 2: plaque_detection.py
        → Plaque detection, PHP/QHPI scoring, report and JSON output

Step 3: plaque_summary.py
        → Clinical-style dashboard image generation
```

---

## Repository Structure

```text
ai-based-dental-plaque-level-determination/
│
├── Data/
│   └── input dental images (.jpg / .jpeg / .png)
│
├── h5/
│   └── mask_rcnn_tooth_single_class_0097.h5
│
├── Teeth Segment Result/
│   └── generated tooth segmentation outputs
│
├── Plaque Result Curves/
│   └── generated plaque analysis outputs
│
├── master_run_pipeline.py
├── Run_Model_1Class.py
├── plaque_detection.py
├── plaque_calculation.py
├── plaque_visualization.py
├── plaque_summary.py
├── generate_brushing_recommendation_from_report.py
├── generate_brushing_recommendation_simple.py
└── README.md
```

The folders `Teeth Segment Result/` and `Plaque Result Curves/` are created automatically when the pipeline runs.

---

## Requirements

This project was developed using:

* Python 3.7.16
* Anaconda / Conda environment
* Mask R-CNN
* OpenCV
* NumPy
* Pillow

Example installation:

```bash
conda create -n MaskRcnn python=3.7.16
conda activate MaskRcnn

pip install numpy pillow opencv-python
```

Mask R-CNN and its required deep learning dependencies must also be installed according to the version used by your trained model and environment.

> The original development environment used a Conda environment named `MaskRcnn`.

---

## Model Weight File

Place the trained Mask R-CNN weight file inside the `h5/` folder:

```text
h5/
└── mask_rcnn_tooth_single_class_0097.h5
```

Expected path:

```text
ai-based-dental-plaque-level-determination/h5/mask_rcnn_tooth_single_class_0097.h5
```

If the weight file is not included in this repository, obtain it separately and place it in the location above before running the pipeline.

---

## Preparing Input Images

Create a folder named `Data` in the repository root:

```text
ai-based-dental-plaque-level-determination/
└── Data/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── ...
```

Place dental photographs inside this folder.

Supported image formats:

```text
.jpg
.jpeg
.png
```

Recommended input characteristics:

* Front-facing image of the upper anterior teeth
* Teeth visible clearly from canine to canine where possible
* Plaque-disclosing solution applied before taking the image
* Adequate lighting with minimal blur
* Avoid images containing personally identifiable information when sharing publicly

---

## Running the Full Pipeline

From the repository root, run:

```bash
python master_run_pipeline.py
```

This executes:

```text
AI Tooth Segmentation
→ Plaque Detection and PHP/QHPI Analysis
→ Summary Dashboard Generation
```

If no previous segmentation output exists, the system automatically runs all three stages.

---

## Optional Pipeline Modes

### Force AI segmentation to run again

```bash
python master_run_pipeline.py --force-ai
```

Use this when input images have changed or when you want to regenerate tooth segmentation outputs.

### Run plaque detection and dashboard only

```bash
python master_run_pipeline.py --detection-only
```

Use this when tooth segmentation outputs already exist and you only want to re-run plaque analysis and dashboard generation.

### Run dashboard generation only

```bash
python master_run_pipeline.py --summary-only
```

Use this when `case_results.json` or plaque analysis reports already exist and you only want to regenerate the dashboard image.

> For real usage with newly captured images, run the full pipeline or use `--force-ai`, because each new dental image requires new tooth segmentation.

---

## Output Files

After processing, the system generates output folders for each case.

Example:

```text
Plaque Result Curves/
└── sample_001/
    ├── case_results.json
    ├── case_summary_php_qhpi.png
    ├── plaque_php_qhpi_report.txt
    │
    ├── tooth11_input_rgb.png
    ├── tooth11_input_rgb_enhanced.png
    ├── tooth11_plaque_mask.png
    ├── tooth11_plaque_vis.png
    ├── tooth11_php_shape_vis.png
    ├── tooth11_qhpi_vis.png
    │
    └── ...
```

Important result files:

| File                         | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `case_summary_php_qhpi.png`  | Dashboard summary of PHP/QHPI results for the case |
| `plaque_php_qhpi_report.txt` | Text report containing tooth-level scores          |
| `case_results.json`          | Structured result data for further processing      |
| `*_plaque_mask.png`          | Detected plaque mask                               |
| `*_plaque_vis.png`           | Plaque overlay visualization                       |
| `*_php_shape_vis.png`        | PHP zone visualization                             |
| `*_qhpi_vis.png`             | QHPI visualization                                 |

---

## Brushing Recommendation Generation

After plaque analysis results have been generated, brushing recommendations can be created from the report files.

Detailed recommendation output:

```bash
python generate_brushing_recommendation_from_report.py
```

Simplified patient-friendly recommendation output:

```bash
python generate_brushing_recommendation_simple.py
```

These scripts generate Thai-language brushing guidance based on PHP/QHPI results and detected plaque locations.

---

## PHP and QHPI Analysis

### PHP Scoring

The system divides each tooth into analysis zones and records whether plaque is present in each zone.

For most teeth, the evaluated zones are:

```text
M = Mesial
I = Incisal
C = Center
G = Gingival
D = Distal
```

For canine teeth viewed from a frontal photograph, the visible surface may be incomplete. The system therefore supports adaptive zone handling for teeth 13 and 23.

### QHPI Scoring

The system estimates QHPI scores using plaque distribution relative to the gingival region and tooth surface coverage.

The current QHPI implementation is a computational approximation designed for research prototyping and should be validated against dentist-provided scores.

---

## Example Result

Add an example input and output image to the repository for easier demonstration:

```text
assets/
├── example_input.jpg
└── example_dashboard.png
```

Example display in README:

```markdown
## Example

### Input Dental Image

![Example Input](assets/example_input.jpg)

### Generated Plaque Analysis Dashboard

![Example Dashboard](assets/example_dashboard.png)
```

> Use only anonymized or authorized example images for public repositories.

---

## Limitations

* The current system focuses on upper anterior teeth only.
* Plaque detection depends on image quality, lighting, and plaque-disclosing color visibility.
* Frontal images may not fully capture the lateral surfaces of canine teeth.
* Pixel-to-millimeter estimation is approximate unless a calibrated reference is available.
* PHP/QHPI scores generated by the system require clinical validation before use in healthcare settings.

---

## Future Development

Potential future improvements include:

* Dentist-validated scoring comparison
* Improved plaque segmentation model training
* Calibration support for physical distance measurement
* Better handling of partially visible teeth
* Mobile or web-based user interface
* Audio-guided feedback for visually impaired users
* Automatic progress tracking across repeated dental photographs

---

## Acknowledgements

Developed during an internship at:

* **PSU-Dolphins Co., Ltd.**

In collaboration with:

* **Faculty of Dentistry, Prince of Songkla University**
* **Faculty of Dentistry, Mahidol University**

---

## Disclaimer

This software is a research prototype and is not a substitute for professional dental examination, diagnosis, or treatment.
