# AI-Powered Crop Health, Soil Conditioning & Pest Risk Monitoring using Multispectral Imaging

A college project that uses aerial multispectral (RGB + Near-Infrared) farmland imagery to assess three things from the same data source: **crop health**, **soil condition**, and **pest risk**.

## Project Overview

This project analyzes aerial farmland images captured in RGB and Near-Infrared (NIR) bands to compute vegetation and moisture indices, then classifies each field image across three dimensions:

1. **Crop Health** — via NDVI (Normalized Difference Vegetation Index)
2. **Soil Condition** — via NDWI (Normalized Difference Water Index) + brightness, as a proxy for moisture/dryness
3. **Pest Risk** — via NDVI patchiness (standard deviation), since pest/weed damage tends to appear as localized patches rather than uniform stress

All three modules are combined into an interactive **Streamlit dashboard**.

## Dataset

We use the **[Agriculture-Vision dataset](https://agriculture-vision.intelinair.com/)** (2019 miniscale subset) — a published aerial farmland image dataset (CVPR 2020) containing:
- 512×512 pixel aerial images
- Separate RGB and NIR image folders (`field_images/rgb`, `field_images/nir`) with matching filenames
- Ground-truth labels for anomalies like weed clusters, planter skips, nutrient deficiency, etc. (not yet used in our current pipeline — see Future Work)

**Note:** The raw dataset (~5GB) is NOT included in this repo (see `.gitignore`). Each team member needs to download it separately from the link above (`data2019_miniscale.tar.gz` + `data2019_splits.json`) and extract it into the project root as `data2019_miniscale/`.

## Methodology

### 1. Crop Health (NDVI)
```
NDVI = (NIR - Red) / (NIR + Red)
```
- NDVI ≥ 0.4 → Healthy
- 0.15 ≤ NDVI < 0.4 → Moderate
- NDVI < 0.15 → Stressed

### 2. Soil Condition (NDWI + brightness proxy)
```
NDWI = (Green - NIR) / (Green + NIR)
```
- NDWI > 0.1 → Moist/Wet
- High brightness + NDWI < -0.1 → Dry/Bare
- Otherwise → Normal

**Limitation:** True soil moisture indices (like NDMI) require a SWIR band, which this dataset doesn't provide. NDWI + brightness is used as an approximation.

### 3. Pest Risk (NDVI patchiness proxy)
```
NDVI standard deviation across the image = "patchiness"
```
- High patchiness + low mean NDVI → High Risk
- Moderate patchiness → Moderate Risk
- Low patchiness → Low Risk

**Limitation:** This is an indirect proxy, not direct pest detection. Real pest identification would need a trained classifier on the dataset's `weed_cluster` labels or dedicated pest imagery.

## Project Structure

```
AI Crop Health/
├── data2019_miniscale/          # dataset (NOT in repo - download separately)
│   └── field_images/
│       ├── rgb/
│       └── nir/
├── venv/                        # virtual environment (NOT in repo)
├── notebook.ipynb                # exploration + module development
├── app.py                        # Streamlit dashboard
├── requirements.txt               # Python dependencies
└── README.md
```

## Setup Instructions (for teammates)

1. **Clone the repo:**
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment (Python 3.11 recommended):**
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Go to https://agriculture-vision.intelinair.com/
   - Download `data2019_miniscale.tar.gz` and `data2019_splits.json`
   - Extract into the project root so you have `data2019_miniscale/field_images/rgb` and `.../nir`

5. **Run the notebook** (for exploration/development):
   ```
   jupyter notebook
   ```

6. **Run the dashboard:**
   ```
   streamlit run app.py
   ```

## Current Status

- ✅ Dataset loading and RGB/NIR band pairing
- ✅ NDVI computation and crop health classification (rule-based)
- ✅ NDWI-based soil condition classification (rule-based)
- ✅ NDVI-patchiness pest risk classification (rule-based)
- ✅ Streamlit dashboard combining all three modules
- ⬜ Trained ML/CNN model (currently rule-based/threshold classification only)
- ⬜ Quantitative validation against ground-truth labels
- ⬜ Final report/documentation

## Known Limitations (be upfront about these in the report/viva)

- Classification uses fixed thresholds, not a trained ML model
- Soil moisture is approximated (no SWIR band available)
- Pest risk is an indirect proxy, not direct pest/disease detection
- Dashboard analyzes one image tile at a time, not a full field map

## Future Work

- Train a CNN or Random Forest classifier using the dataset's existing anomaly labels (e.g., predict `weed_cluster` presence) for a real ML component
- Add quantitative accuracy/confusion matrix evaluation
- Stitch multiple tiles into a full-field map view

## Team

- [Add team member names and roles here]
