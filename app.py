"""
app.py — Flask backend for CropSense AI
Crop Prediction & Yield Production using Random Forest models.

Run with:
    pip install flask scikit-learn
    python app.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle, json, numpy as np, os

app = Flask(__name__, static_folder='frontend', static_url_path='')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

print("Loading models...")
with open(os.path.join(MODELS_DIR, 'crop_classifier.pkl'), 'rb') as f:
    classifier = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'production_regressor.pkl'), 'rb') as f:
    regressor = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'label_encoder_crop.pkl'), 'rb') as f:
    le_crop = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'label_encoder_season.pkl'), 'rb') as f:
    le_season = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'label_encoder_state.pkl'), 'rb') as f:
    le_state = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'label_encoder_district.pkl'), 'rb') as f:
    le_district = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'agg_features.json')) as f:
    agg = json.load(f)
with open(os.path.join(MODELS_DIR, 'state_district_map.json')) as f:
    state_district_map = json.load(f)
with open(os.path.join(MODELS_DIR, 'meta.json')) as f:
    meta = json.load(f)
print("All models loaded. Ready.")


def build_features(temp, humidity, soil_moisture, area, state_name, district_name, season, production_estimate):
    log_area   = np.log1p(area)
    log_prod   = np.log1p(production_estimate)
    yield_val  = production_estimate / (area + 1)
    log_yield  = np.log1p(yield_val)
    sqrt_area  = np.sqrt(area)
    temp_humidity     = temp * humidity
    temp_moisture     = temp * soil_moisture
    humidity_moisture = humidity * soil_moisture
    ds_key = "('{}', '{}')".format(district_name, season)
    ss_key = "('{}', '{}')".format(state_name, season)
    district_season_yield  = agg['district_season_yield'].get(ds_key,  agg['global_yield_median'])
    state_season_yield     = agg['state_season_yield'].get(ss_key,     agg['global_yield_median'])
    district_crop_freq     = agg['district_crop_freq'].get(ds_key,     50.0)
    district_yield_mean    = agg['district_yield_mean'].get(district_name, agg['global_yield_median'])
    district_yield_std     = agg['district_yield_std'].get(district_name, 0.0)
    state_yield_median     = agg['state_yield_median'].get(state_name, agg['global_yield_median'])
    season_area_median     = agg['season_area_median'].get(season,     agg['global_area_median'])
    yield_vs_district_mean = yield_val / (district_yield_mean + 1)
    crop_rank_ds           = 1.0
    state_enc    = le_state.transform([state_name])[0]
    district_enc = le_district.transform([district_name])[0]
    season_enc   = le_season.transform([season])[0]
    crop_year_placeholder = 2010  # neutral median; removed from UI
    return np.array([[
        temp, humidity, soil_moisture, area,
        log_area, sqrt_area,
        log_prod, yield_val, log_yield,
        temp_humidity, temp_moisture, humidity_moisture,
        district_season_yield, state_season_yield,
        district_crop_freq, crop_rank_ds,
        district_yield_mean, district_yield_std,
        state_yield_median, season_area_median, yield_vs_district_mean,
        state_enc, season_enc, district_enc, crop_year_placeholder
    ]])


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        temp          = float(data['temperature'])
        humidity      = float(data['humidity'])
        soil_moisture = float(data['soil_moisture'])
        area          = float(data['area'])
        state_name    = str(data['state']).strip()
        district_name = str(data['district']).strip().upper()
        season        = str(data['season']).strip()
        production_est = agg['global_production_median']

        if state_name not in le_state.classes_:
            return jsonify({'error': f"Unknown state: {state_name}"}), 400
        if district_name not in le_district.classes_:
            return jsonify({'error': f"Unknown district: {district_name}"}), 400
        if season not in le_season.classes_:
            return jsonify({'error': f"Unknown season: {season}"}), 400

        X = build_features(temp, humidity, soil_moisture, area,
                           state_name, district_name, season, production_est)

        # Crop Prediction
        crop_encoded = classifier.predict(X)[0]
        best_crop    = le_crop.inverse_transform([crop_encoded])[0]
        proba        = classifier.predict_proba(X)[0]
        top_indices  = proba.argsort()[-5:][::-1]
        top3 = [[le_crop.inverse_transform([i])[0], round(float(proba[i]), 4)]
                for i in top_indices[:3]]

        # Yield Production
        production   = max(0.0, float(regressor.predict(X)[0]))
        yield_per_ha = production / area if area > 0 else 0

        return jsonify({
            'best_crop'   : best_crop,
            'production'  : round(production, 2),
            'yield_per_ha': round(yield_per_ha, 2),
            'top3'        : top3,
            'model'       : 'Random Forest Classifier',
            'accuracy'    : '85%+',
            'inputs': {
                'temperature'  : temp,
                'humidity'     : humidity,
                'soil_moisture': soil_moisture,
                'area'         : area,
                'state'        : state_name,
                'district'     : district_name,
                'season'       : season,
            }
        })
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/meta', methods=['GET'])
def get_meta():
    return jsonify({
        'states'             : meta['states'],
        'seasons'            : meta['seasons'],
        'state_district_map' : state_district_map
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest', 'accuracy': '85%+'})


if __name__ == '__main__':
    print()
    print("=" * 55)
    print("  CropSense AI — Flask Backend")
    print("  Crop Prediction + Yield Production")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
