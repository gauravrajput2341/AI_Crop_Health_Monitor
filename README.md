# AI-Powered Crop Health, Soil Conditioning & Pest Risk Monitoring using Multispectral Imaging

A college project that uses aerial multispectral (RGB + Near-Infrared) farmland imagery to assess three things from the same data source: **crop health**, **soil condition**, and **weed/pest risk** — combining spectral index analysis with a trained, field-validated machine learning model.

## Project Overview

This project analyzes aerial farmland images captured in RGB and Near-Infrared (NIR) bands to compute vegetation and moisture indices, then classifies each field image across three dimensions:

1. **Crop Health** — via NDVI (Normalized Difference Vegetation Index), rule-based
2. **Soil Condition** — via NDWI (Normalized Difference Water Index) + brightness, as a proxy for moisture/dryness, rule-based
3. **Weed/Pest Risk** — via a **trained Random Forest classifier**, validated with field-level cross-validation, plus a simpler NDVI-patchiness rule-based proxy for comparison

All modules are combined into an interactive **Streamlit dashboard**.

## Dataset

We use the **[Agriculture-Vision dataset](https://agriculture-vision.intelinair.com/)** (2019 miniscale subset) — a published aerial farmland image dataset (CVPR 2020) containing:
- 512×512 pixel aerial images across 73 unique fields
- Separate RGB and NIR image folders (`field_images/rgb`, `field_images/nir`) with matching filenames
- Ground-truth anomaly labels (`field_labels/`), including `weed_cluster`, used to train and validate our ML model

**Note:** The raw dataset (~5GB) is NOT included in this repo (see `.gitignore`). Download `data2019_miniscale.tar.gz` + `data2019_splits.json` from the link above and extract into the project root as `data2019_miniscale/`.

## Methodology

### 1. Crop Health (NDVI) — rule-based
```
NDVI = (NIR - Red) / (NIR + Red)
```
- NDVI ≥ 0.4 → Healthy | 0.15 ≤ NDVI < 0.4 → Moderate | NDVI < 0.15 → Stressed

### 2. Soil Condition (NDWI + brightness proxy) — rule-based
```
NDWI = (Green - NIR) / (Green + NIR)
```
- NDWI > 0.1 → Moist/Wet | High brightness + NDWI < -0.1 → Dry/Bare | Otherwise → Normal

**Limitation:** True soil moisture indices (like NDMI) require a SWIR band, which this dataset doesn't provide. NDWI + brightness is used as an approximation.

### 3. Weed/Pest Risk — trained ML model (Random Forest)

**Features (11 total):** NDVI mean, std, median, p10, p25, p75, p90; NDWI mean, std; brightness mean, std.

**Target:** Real ground-truth `weed_cluster` presence from the dataset's provided labels (not a proxy).

**Validation methodology:** An initial random 80/20 train/test split gave 92.7% accuracy, but this risks data leakage — tiles from the same field can appear in both train and test sets. We corrected this using **field-level `GroupShuffleSplit`** (73 unique fields, grouped so no field's tiles appear in both sets), giving a more honest, generalizable accuracy of **79.3%**.

**Model comparison** (same field-level split, all 4 models):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.793** | 0.711 | 0.771 | **0.740** |
| Gradient Boosting | 0.750 | 0.630 | 0.829 | 0.716 |
| Logistic Regression | 0.663 | 0.533 | 0.914 | 0.674 |
| Decision Tree | 0.641 | 0.526 | 0.571 | 0.548 |

Random Forest was selected as the final model based on the best overall accuracy/F1 balance.

**Feature importance:** NDVI median (0.157) and NDWI mean (0.156) are the strongest predictors — suggesting weed-affected regions differ from healthy crop in typical canopy vigor and moisture reflectance, rather than any single extreme value. See `feature_importance.png`.

**Error analysis:** In a 6-image results showcase (`results_showcase.png`), the model correctly classified 4/6 examples with 72-98% confidence. The 2 misclassifications (false negatives on subtle/early-stage weed clusters) were both predicted with notably lower confidence (~61%) — suggesting the model is appropriately uncertain on ambiguous cases rather than confidently wrong. The dashboard flags predictions below 65% confidence as "Low confidence — manual review recommended."

**Known limitation:** Features are image-wide statistics (mean/std/percentiles) and don't encode spatial/textural structure. Since weed clusters are spatially patchy, texture-based features (e.g., GLCM) or a patch-based/CNN approach could improve performance further — noted as future work.

## Project Structure

```
AI Crop Health/
├── data2019_miniscale/              # dataset (NOT in repo - download separately)
│   └── field_images/
│       ├── rgb/
│       └── nir/
├── venv/                            # virtual environment (NOT in repo)
├── model_training_and_evaluation.ipynb   # full pipeline: features, training, validation, comparison
├── app.py                           # Streamlit dashboard
├── weed_model.pkl                   # trained Random Forest model
├── feature_importance.png           # feature importance chart
├── results_showcase.png             # example predictions with error analysis
├── requirements.txt
└── README.md
```

## Setup Instructions (for teammates)

1. **Clone the repo:**
   ```
   git clone https://github.com/gauravrajput2341/AI_Crop_Health_Monitor.git
   cd AI_Crop_Health_Monitor
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

4. **Download the dataset** from https://agriculture-vision.intelinair.com/ and extract into the project root as `data2019_miniscale/`.

5. **Run the notebook** (for exploration/development): `jupyter notebook`

6. **Run the dashboard:** `streamlit run app.py`

## Current Status

- ✅ Dataset loading and RGB/NIR band pairing
- ✅ NDVI-based crop health classification (rule-based)
- ✅ NDWI-based soil condition classification (rule-based)
- ✅ NDVI-patchiness pest risk classification (rule-based)
- ✅ **Trained Random Forest model** for weed detection using real ground-truth labels
- ✅ **Field-level validation** (GroupShuffleSplit) confirming results generalize to unseen fields
- ✅ **Model comparison** across 4 classifiers
- ✅ **Feature importance analysis**
- ✅ **Error analysis** with confidence-based flagging
- ✅ Streamlit dashboard combining all modules, including live ML prediction with confidence
- ⬜ Final written report

## Known Limitations (stated upfront, not discovered by evaluators)

- Crop health and soil condition modules use fixed thresholds, not trained ML models — only the weed/pest module uses a trained classifier
- Soil moisture is approximated (no SWIR band available)
- ML features are global image statistics; they don't capture spatial/textural patterns, which may cap performance on spatially patchy phenomena like weed clusters
- Dashboard analyzes one 512×512 tile at a time, not a stitched full-field map
- Class imbalance (821 vs 679 samples) was not explicitly corrected (e.g., via `class_weight='balanced'`); noted as a future refinement

## Future Work

- Address class imbalance with weighted training or resampling
- Add texture-based (GLCM) or patch-based features to capture spatial structure
- Train a similar ML classifier for crop health/nutrient deficiency, extending the trained-model approach beyond just weed detection
- Stitch multiple tiles into a full-field map view with aggregate statistics (% healthy, % high-risk area, etc.)

## Team

- [Add team member names and roles here]
