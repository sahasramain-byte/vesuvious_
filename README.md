# Vesuvius Surface Detection Demo

Detects papyrus surfaces in 3D CT scans of ancient Herculaneum scrolls using a 3D UNet deep learning model. Built for the Vesuvius Challenge Kaggle competition.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Works immediately with built-in synthetic data — no uploads required.

---

## How it works

### The problem
Ancient scrolls were buried in volcanic ash and can't be physically unrolled. CT scans create 3D volumes of the scrolls, and the goal is to detect the papyrus surface layers inside those volumes.

### The model (`model.py`)
A **3D UNet** — an encoder-decoder neural network with skip connections.
- Takes a 3D CT volume as input
- Outputs a binary mask: which voxels are papyrus surface vs. background
- ~2.7M parameters

### Inference (`inference.py`)
The CT volumes are too large to process all at once, so **sliding window inference** is used:
- Breaks the volume into overlapping 64³ patches
- Runs each patch through the model
- Stitches the results back together (averaging overlaps)

### Synthetic data (`synthetic.py`)
Generates fake-but-realistic CT volumes for demo purposes:
- Simulated papyrus sheets as curved sinusoidal surfaces
- Realistic background noise and texture

### Metrics & visualization (`utils.py`)
- **Dice / IoU / Precision / Recall** to evaluate predictions
- Slice views, overlays, probability histograms, and 3D surface plots

---

## App pages

| Page | What it shows |
|---|---|
| Overview | Background on the challenge and pipeline |
| Data Explorer | Slice through the CT volume (XY, XZ, YZ) |
| Model Architecture | Layer-by-layer diagram of the UNet |
| Inference & Results | CT / Ground Truth / Prediction / Overlay |
| Metrics & Analysis | Scores + 3D surface visualization |

---

## Using real data

1. **Data Explorer** — upload a `.tif` CT volume
2. **Inference & Results** — upload model weights (`.pth`)

The app will run live inference on your data instead of the synthetic demo.

---

## File structure

```
vesuvius-app/
├── app.py            # UI and page routing
├── model.py          # 3D UNet definition
├── inference.py      # Sliding window inference
├── synthetic.py      # Demo data generation
├── utils.py          # Metrics + visualization
└── requirements.txt
```
