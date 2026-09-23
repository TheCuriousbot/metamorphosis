# Cast Iron ISO 945 Analyzer V6

ISO 945-4:2019-aligned image-analysis prototype for spheroidal/nodular graphite cast iron.

## V6 improvements over V5
- Fast multi-scale processing: large microscope images are automatically downscaled with calibration adjusted, preventing long hangs on high-resolution files.
- AUTO stable segmentation tests public/reference Otsu, enhanced Otsu/local rescue, and blackhat local-darkness rescue.
- AUTO separation tests peak ratios 0.50/0.60/0.70 and selects using internal fragmentation diagnostics; manual mode remains available.
- Real-reference calibration search against a trusted known nodularity.
- Multi-field analysis and ISO-style PDF report with original image, mask, particle overlay, measurements and sampling status.
- Particle CSV export.

## ISO basis
ISO 945-4:2019 is the current edition (confirmed 2024). The prototype follows the image-analysis concepts of maximum Feret diameter, roundness and exclusion of small/border-intersecting particles. It is **not** an accredited or certified ISO conformity system.

## Run
```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Recommended first settings
- Engine: **AUTO — stable multi-method**
- Minimum Feret: **10 µm**
- Separate joined particles: ON
- Separation: **AUTO**
- Maximum analysis dimension: **1800 px**

If you know your microscope calibration, enter µm/pixel. If not, the app can still produce a screening nodularity/particle analysis, but calibrated size and particles/mm² should not be treated as metrological results.

## Data for calibration/validation
Best input: 5–15 real microscope fields with trusted nodularity, magnification/calibration or field width, independent particle count if available, and difficult-case notes. A few manually corrected particle masks are especially valuable.
