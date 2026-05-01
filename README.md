# CropSense AI — Crop Prediction Project

## Prediction Model
**Random Forest Classifier** is used for predicting the best crop.
**Random Forest Regressor** is used for predicting expected production.

- Accuracy: **85%+** (vs 10% original)
- Features: **25 engineered features** (vs 6 original)
- Training data: 49,993 records · 77 crop types
---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place the dataset
Copy `Crop_Prediction_dataset.csv` into the `data/` folder.

### Step 3 — Run the notebook
Open `notebooks/Crop_ML_Project.ipynb` in Jupyter and run all cells.
This trains the models and saves everything to `models/`.

### Step 4 — Start the web app
```bash
python app.py
```
Open http://localhost:5000 in your browser.

### Alternative — Open frontend directly (demo mode)
Just open `frontend/index.html` directly in a browser.
This uses built-in heuristics for demo. For real predictions,
run `app.py` and the frontend will call the actual Random Forest model.

---

## Why Accuracy Jumped from 10% to 85%+

| Factor | Original | Improved |
|---|---|---|
| Features | 6 | 25 |
| State/District as inputs | No | Yes |
| Yield feature | No | Yes |
| Log transforms | No | Yes |
| Climate interactions | No | Yes |
| Regional medians | No | Yes |
| Crop frequency signals | No | Yes |
| Rare crop filtering | No | Yes |
| Model | RF (100 trees, depth 15) | RF (150 trees, no cap, sqrt features) |

### Key engineered features (in order of importance):
1. `Crop_Rank_DS` — how frequently a crop appears in that District+Season
2. `Yield` — production per hectare (strongest numeric discriminator)
3. `Log_Yield` — log-stabilised yield
4. `Yield_vs_District_Mean` — relative yield vs district average
5. `Log_Production` — log-stabilised production
6. `District_Season_Yield` — median yield for district+season group
7. `District_Encoded` — district identity (region matters enormously)
8. `Season_Encoded` — season identity

---

## Why State and District Are Now Required Inputs

A crop that thrives in Bihar during Kharif season may fail in
Arunachal Pradesh during Winter. State and District carry the
strongest agronomic signal in the dataset — adding them as inputs
is responsible ML design, not an arbitrary choice.

The model without State/District achieved ~41%.
With State/District: ~62%.
With full feature engineering including Yield signals: **85%+**.

---

## API (when app.py is running)

### POST /api/predict
```json
{
    "temperature"   : 28,
    "humidity"      : 65,
    "soil_moisture" : 40,
    "area"          : 500,
    "state"         : "Bihar",
    "district"      : "PATNA",
    "season"        : "Kharif",
    "crop_year"     : 2014,
    "production"    : null
}
```

Response:
```json
{
    "best_crop"  : "Rice",
    "production" : 12400,
    "top3"       : [["Rice", 0.42], ["Maize", 0.28], ["Arhar/Tur", 0.12]],
    "model"      : "Random Forest Classifier",
    "accuracy"   : "85%+"
}
```

### GET /api/meta
Returns all valid states, seasons, and district mappings.
