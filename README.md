# Espresso Analysis

> A machine-learning-assisted tool for recording, evaluating, and improving 
> your espresso shots. Built around the Varia AKU Mini scale and a 1D CNN 
> shot classifier, and a Gaussian Process grind-size optimizer.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & Customization](#configuration--customization)
- [Project Structure](#project-structure)
- [Model Details](#model-details)

---

## Overview

**Espresso Analysis** logs and analyzes every espresso shot you pull, via 
streamed data from the **Varia AKU Mini** smart scale. It uses that data to:
1. **Classify shot quality** (under, balanced, or an over extraction) with a 
   1D CNN trained on your shot history.
2. **Recommends your next grind setting** using a Gaussian Process 
   Optimization model that learns the relationship between grind size and 
   extraction quality.

It was originally built for my **Gaggia Classic Pro E24** espresso machine, 
but can be used with any espresso machine provided you have a **Varia AKU Mini**
scale to record shot data.

> **Note:** The on-screen display was designed for a vintage 20" CRT, 
> specifically the [1993 RCA X20101GS TV](https://newspapers.digitalnc.org/lccn/sn92068245/1993-08-25/ed-1/seq-12/),
> so the UI may not display correctly on modern monitors.

## How It Works

```
   Varia AKU Mini (BLE)
          │  live weight stream
          ▼
   BLE_logger.py / log_shot.py   ──►  shots/*.csv  (per-shot weight curves)
          │
          ▼
   labeling_handler.py  ──►  shots/manifest.csv (labels + shot metadata)
          │
          ├──► model.py            (1D CNN: predicts shot quality)
          │
          └──► recommend_grind.py  (Gaussian Process: recommends next grind)
          │
          ▼
   display.py  ──►  on-screen results + next-shot guidance
```

`manager.py` coordinates the project, calling the other modules in the 
correct order and retrains both models when new shots are logged.

## Hardware Requirements

- An espresso machine (e.g., Gaggia Classic Pro E24)
- A **Varia AKU Mini** smart scale (BLE-enabled)
- A computer with BLE support (e.g., Raspberry Pi, laptop, etc.)
- *Optional:* A small display/monitor for on-screen feedback

## Installation

```bash
# Clone repository
git clone https://github.com/arvickc/Espresso-Analysis.git
cd Espresso-Analysis

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Confirm that your device can connect to the scale:
```bash
python debug_connection.py
```

## Usage

```bash
python manager.py
```

1. The app boots to an idle screen, press `Enter` to continue
2. Follow the prompts to enter shot metadata (e.g., dose, grind-size, etc.)
3. Start pulling your shot, the scale will stream weight data to the app
4. After the shot, the app will prompt you to classify the shot and show the 
   next grind recommendation based on your previous shots.
5. The app will enter idle mode again, ready for the next shot.

### Cold Start

The CNN needs a minimum of **15 labeled shots** before it can train:
- 5 balanced
- 5 under-extracted
- 5 over-extracted

Until then, the app still logs and labels shots, but will not provide shot 
quality information or grind recommendations.

## Configuration & Customization

Create a `config.py` file in the root directory set to your defaults:

```python
KNOWN_ADDRESS = "AA:BB:CC:DD:EE:FF" # Connect directly to scale
GRIND_SETTING = "3.0"
DOSE_G = "18.0"
BEAN_NAME = "house_blend"
ROAST_DATE = ""
OPEN_DATE = ""
```

Other configurable parameters:

| What                       | Where                              | Notes                                                      |
|----------------------------|------------------------------------|------------------------------------------------------------|
| Shot quality labels        | `preprocessing.py` (`LABELS`)      | Add/rename classes beyond under/balanced/over              |
| CNN architecture           | `model.py` (`ShotCNN`)             | Swap conv layers, channels, or classifier head             |
| Retraining threshold       | `model.py` (`MIN_SHOTS_PER_CLASS`) | Require more/fewer shots per class before retraining       |
| Grind recommendation logic | `recommend_grind.py`               | Adjust the Gaussian Process kernel/acquisition strategy    |
| Display / UI               | `display.py`                       | Swap the CRT-tuned layout for a modern monitor, add themes |
| Per-bean tracking          | `manager.py`                       | Grind models are already saved per-bean under `gp_models/` |

## Project Structure
 
```
Espresso-Analysis/
├── manager.py             # Main orchestrator / entry point
├── BLE_logger.py          # Bluetooth scale discovery + connection
├── log_shot.py            # Live shot streaming/logging session
├── labeling_handler.py    # Shot labeling + manifest management
├── preprocessing.py       # Curve resampling, feature extraction, LABELS
├── model.py               # ShotCNN definition, training, inference
├── recommend_grind.py     # Gaussian Process grind recommender
├── display.py             # On-screen UI / app state machine
├── debug_connection.py    # Standalone BLE connection tester
├── assets/                # UI assets
├── shots/                 # Per-shot CSVs + manifest.csv (generated)
└── gp_models/             # Saved per-bean GP models (generated)
```

## Model Details

**Shot Classifier - `model.py`**
- 1D CNN over the (time-aligned) weight/flow-rate curve of a shot
- 3 convolutional blocks (16 -> 32 -> 64 channels) with batch normalization 
  ReLU activate, global average pooling, and a linear classifier head
- Trained with an 80/20 train/val split, Adam optimizer, and cross-entropy loss
- Retrained automatically whenever a new non-discarded shot is added

**Grind Recommender - `recommend_grind.py`**
- Learns a function from grind setting -> extraction outcome
- Recommends the next grind setting to try, with model state saved per-bean 
  in `gp_models/`
