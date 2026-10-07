# ============================================================



# AGRIWATCH BACKEND API



# Drought Prediction System



# ============================================================







import os
import re
import time



import json



import joblib



import numpy as np



import pandas as pd







from fastapi import FastAPI, HTTPException



from fastapi.middleware.cors import CORSMiddleware



from pydantic import BaseModel



from typing import Dict











# ============================================================



# 1. FASTAPI APPLICATION



# ============================================================







app = FastAPI(



    title="AgriWatch API",



    description="Machine Learning Drought Prediction API for AgriWatch",



    version="1.0.0"



)











# ============================================================



# 2. CORS



# ============================================================







app.add_middleware(



    CORSMiddleware,



    # Allow local development and the deployed frontend during initial testing.

    # After the production frontend URL is finalized, this can be restricted.

    allow_origins=["*"],



    allow_credentials=False,



    allow_methods=["*"],



    allow_headers=["*"],



)











# ============================================================



# 3. PROJECT PATHS



# ============================================================







BASE_DIR = os.path.dirname(os.path.abspath(__file__))







MODEL_PATH = os.path.join(



    BASE_DIR,



    "model",



    "agriwatch_drought_model.pkl"



)







CONFIG_PATH = os.path.join(



    BASE_DIR,



    "model",



    "agriwatch_model_config.json"



)







DATA_PATH = os.path.join(



    BASE_DIR,



    "data",



    "AgriWatch_Master_Dataset_2015_2026.csv"



)











# ============================================================



# 4. VERIFY REQUIRED FILES



# ============================================================







if not os.path.exists(MODEL_PATH):



    raise FileNotFoundError(



        f"Model file not found:\n{MODEL_PATH}"



    )







if not os.path.exists(CONFIG_PATH):



    raise FileNotFoundError(



        f"Model configuration file not found:\n{CONFIG_PATH}"



    )







if not os.path.exists(DATA_PATH):



    raise FileNotFoundError(



        f"Master dataset not found:\n{DATA_PATH}"



    )











# ============================================================



# 5. LOAD TRAINED MODEL



# ============================================================







print("=" * 60)



print("LOADING AGRIWATCH ML SYSTEM")



print("=" * 60)







model = joblib.load(MODEL_PATH)







print("✓ ML model loaded successfully")











# ============================================================



# 6. LOAD MODEL CONFIGURATION



# ============================================================







with open(CONFIG_PATH, "r") as file:



    loaded_config = json.load(file)







if "features" not in loaded_config:



    raise ValueError(



        "Model config does not contain 'features'."



    )







if "warning_threshold" not in loaded_config:



    raise ValueError(



        "Model config does not contain 'warning_threshold'."



    )







MODEL_FEATURES = loaded_config["features"]







WARNING_THRESHOLD = float(
    loaded_config["warning_threshold"]
)

# The production model/config must agree on the exact ordered feature set.
if len(MODEL_FEATURES) != 24:
    raise ValueError(
        f"Expected the upgraded 24-feature model config, found {len(MODEL_FEATURES)} features."
    )

required_model_features = {
    "rainfall_mm",
    "soil_moisture",
    "ndvi",
    "temperature_c",
    "evaporation_mm",
    "rainfall_3month",
    "spi3",
    "rainfall_mm_lag1",
    "rainfall_mm_lag2",
    "rainfall_mm_lag3",
    "soil_moisture_lag1",
    "soil_moisture_lag2",
    "soil_moisture_lag3",
    "ndvi_lag1",
    "ndvi_lag2",
    "ndvi_lag3",
    "temperature_c_lag1",
    "temperature_c_lag2",
    "temperature_c_lag3",
    "evaporation_mm_lag1",
    "evaporation_mm_lag2",
    "evaporation_mm_lag3",
    "month_sin",
    "month_cos",
}

missing_model_features = required_model_features - set(MODEL_FEATURES)
if missing_model_features:
    raise ValueError(
        "Model config is missing upgraded features: "
        + ", ".join(sorted(missing_model_features))
    )

model_feature_count = getattr(model, "n_features_in_", None)
if model_feature_count is not None and int(model_feature_count) != len(MODEL_FEATURES):
    raise ValueError(
        f"Saved model expects {model_feature_count} features but config contains {len(MODEL_FEATURES)}."
    )

model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None and list(model_feature_names) != list(MODEL_FEATURES):
    raise ValueError(
        "Saved model feature order does not match agriwatch_model_config.json."
    )







print("✓ Model configuration loaded")



print("  Number of model features:", len(MODEL_FEATURES))



print("  Warning threshold:", WARNING_THRESHOLD)











# ============================================================



# 7. LOAD MASTER ENVIRONMENTAL DATASET



# ============================================================







master_df = pd.read_csv(DATA_PATH)







required_columns = [
    "date",
    "province",
    "district",
    "rainfall_mm",
    "rainfall_3month",
    "spi3",
    "soil_moisture",
    "ndvi",
    "temperature_c",
    "evaporation_mm",
]







missing_columns = [



    column



    for column in required_columns



    if column not in master_df.columns



]







if missing_columns:



    raise ValueError(



        "Master dataset is missing columns: "



        + ", ".join(missing_columns)



    )







master_df["date"] = pd.to_datetime(



    master_df["date"],



    errors="coerce"



)







master_df = master_df.dropna(



    subset=["date", "district"]



)







master_df["district"] = (



    master_df["district"]



    .astype(str)



    .str.strip()



)







master_df["province"] = (
    master_df["province"]
    .astype(str)
    .str.strip()
)

numeric_columns = [
    "rainfall_mm",
    "rainfall_3month",
    "spi3",
    "soil_moisture",
    "ndvi",
    "temperature_c",
    "evaporation_mm",
]

for column in numeric_columns:
    master_df[column] = pd.to_numeric(
        master_df[column],
        errors="coerce"
    )







master_df = (



    master_df



    .sort_values(["district", "date"])



    .reset_index(drop=True)



)







print("✓ Master dataset loaded successfully")



print("  Records:", len(master_df))



print("  Districts:", master_df["district"].nunique())



print(



    "  Date range:",



    master_df["date"].min(),



    "to",



    master_df["date"].max()



)







print("=" * 60)











# ============================================================



# 8. REQUEST MODEL FOR MANUAL /PREDICT ENDPOINT



# ============================================================







class PredictionRequest(BaseModel):



    features: Dict[str, float]











# ============================================================



# 9. RISK LEVEL FUNCTION



# ============================================================







def calculate_risk_level(probability: float):







    if probability < 0.25:



        return "Low"







    elif probability < 0.50:



        return "Moderate"







    elif probability < 0.65:



        return "High"







    else:



        return "Severe"











# ============================================================



# 10. CREATE MODEL DATAFRAME



# ============================================================







def create_model_input(features: dict):







    missing_features = [



        feature



        for feature in MODEL_FEATURES



        if feature not in features



    ]







    if missing_features:



        raise ValueError(



            "Missing model features: "



            + ", ".join(missing_features)



        )







    ordered_values = [



        float(features[feature])



        for feature in MODEL_FEATURES



    ]







    input_df = pd.DataFrame(



        [ordered_values],



        columns=MODEL_FEATURES



    )







    return input_df











# ============================================================



# 11. RUN MODEL PREDICTION



# ============================================================







def run_prediction(features: dict):







    input_df = create_model_input(features)







    probability = float(



        model.predict_proba(input_df)[0][1]



    )







    predicted_drought = int(



        probability >= WARNING_THRESHOLD



    )







    prediction_status = (



        "Drought Warning"



        if predicted_drought == 1



        else "No Drought"



    )







    risk_level = calculate_risk_level(



        probability



    )







    return {



        "success": True,







        "drought_probability": round(



            probability,



            4



        ),







        "drought_probability_percent": round(



            probability * 100,



            2



        ),







        "predicted_drought": predicted_drought,







        "prediction_status": prediction_status,







        "risk_level": risk_level,







        "warning_threshold": WARNING_THRESHOLD,



    }











# ============================================================



# 12. DISTRICT NAME MATCHING



# ============================================================







def find_district(district_name: str):
    """Find a district while remaining compatible with old frontend labels.

    The upgraded 160-district master uses cleaner/current names such as
    "Jacobabad", while older frontend routes/data may still send
    "Jacobabad District" or agency-style labels.  Matching therefore uses
    both the exact name and a normalized name with administrative suffixes
    and punctuation removed.
    """

    requested_raw = str(district_name or "").strip()
    if not requested_raw:
        raise ValueError("District name is required.")

    def normalize_name(value):
        value = str(value or "").strip().lower()
        # Keep old frontend/old-boundary names compatible with the new master.
        value = re.sub(r"\b(district|agency)\b", "", value)
        value = re.sub(r"[^a-z0-9]+", "", value)
        return value

    requested_lower = requested_raw.lower()

    # Exact match first.
    exact = master_df[
        master_df["district"].str.lower() == requested_lower
    ]
    if not exact.empty:
        return exact.copy()

    requested_key = normalize_name(requested_raw)

    # Common spelling/name aliases that normalization alone may not solve.
    alias_keys = {
        "deraismailkhan": "dikhan",
        "dikh": "dikhan",
        "kambershahdadkot": "qambarshahdadkot",
        "qambershahdadkot": "qambarshahdadkot",
        "shaheedbenazirabad": "shaheedbenazirabad",
        "nawabshah": "shaheedbenazirabad",
        "torghar": "torghar",
        "torgharh": "torghar",
    }
    requested_key = alias_keys.get(requested_key, requested_key)

    district_names = master_df["district"].drop_duplicates().tolist()
    normalized_matches = []

    for name in district_names:
        key = normalize_name(name)
        key = alias_keys.get(key, key)
        if key == requested_key:
            normalized_matches.append(name)

    if len(normalized_matches) == 1:
        return master_df[
            master_df["district"] == normalized_matches[0]
        ].copy()

    # Conservative partial normalized match as a final fallback.
    partial_matches = []
    for name in district_names:
        key = alias_keys.get(normalize_name(name), normalize_name(name))
        if requested_key and (requested_key in key or key in requested_key):
            partial_matches.append(name)

    partial_matches = sorted(set(partial_matches))

    if len(partial_matches) == 1:
        return master_df[
            master_df["district"] == partial_matches[0]
        ].copy()

    if len(partial_matches) > 1:
        raise ValueError(
            "Multiple districts matched. Matches: "
            + ", ".join(partial_matches)
        )

    raise ValueError(
        f"District '{district_name}' was not found."
    )











# ============================================================



# 13. BUILD 24 FEATURES AUTOMATICALLY



# ============================================================







def build_district_features(district_name: str):
    """Build the exact ordered inputs required by the upgraded 24-feature RF."""

    district_df = (
        find_district(district_name)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Current month + previous 3 months are required by the lag features.
    if len(district_df) < 4:
        raise ValueError(
            f"Not enough historical data for '{district_name}'."
        )

    current = district_df.iloc[-1]
    lag1 = district_df.iloc[-2]
    lag2 = district_df.iloc[-3]
    lag3 = district_df.iloc[-4]

    month = int(current["date"].month)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = {
        # Current environmental conditions
        "rainfall_mm": float(current["rainfall_mm"]),
        "soil_moisture": float(current["soil_moisture"]),
        "ndvi": float(current["ndvi"]),
        "temperature_c": float(current["temperature_c"]),
        "evaporation_mm": float(current["evaporation_mm"]),

        # Added for the upgraded next-month model
        "rainfall_3month": float(current["rainfall_3month"]),
        "spi3": float(current["spi3"]),

        # Rainfall lags
        "rainfall_mm_lag1": float(lag1["rainfall_mm"]),
        "rainfall_mm_lag2": float(lag2["rainfall_mm"]),
        "rainfall_mm_lag3": float(lag3["rainfall_mm"]),

        # Soil moisture lags
        "soil_moisture_lag1": float(lag1["soil_moisture"]),
        "soil_moisture_lag2": float(lag2["soil_moisture"]),
        "soil_moisture_lag3": float(lag3["soil_moisture"]),

        # NDVI lags
        "ndvi_lag1": float(lag1["ndvi"]),
        "ndvi_lag2": float(lag2["ndvi"]),
        "ndvi_lag3": float(lag3["ndvi"]),

        # Temperature lags
        "temperature_c_lag1": float(lag1["temperature_c"]),
        "temperature_c_lag2": float(lag2["temperature_c"]),
        "temperature_c_lag3": float(lag3["temperature_c"]),

        # Evaporation lags
        "evaporation_mm_lag1": float(lag1["evaporation_mm"]),
        "evaporation_mm_lag2": float(lag2["evaporation_mm"]),
        "evaporation_mm_lag3": float(lag3["evaporation_mm"]),

        # Seasonal encoding
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
    }

    # Only the features listed by the saved config are sent to the model.
    # This keeps the model order identical to training.
    missing_features = [
        feature for feature in MODEL_FEATURES
        if feature not in features
    ]
    if missing_features:
        raise ValueError(
            "Unable to build configured model features: "
            + ", ".join(missing_features)
        )

    invalid_features = [
        feature for feature in MODEL_FEATURES
        if pd.isna(features[feature]) or not np.isfinite(float(features[feature]))
    ]
    if invalid_features:
        raise ValueError(
            "Missing environmental values for "
            f"{district_name}: "
            + ", ".join(invalid_features)
        )

    return features, current












# ============================================================
# FAST LATEST-PREDICTION CACHE
# ============================================================
#
# Why this exists:
# The old /map-data and /compare-districts endpoints called
# predict_district() 160 separate times. With a 500-tree
# Random Forest that creates a lot of repeated Python/sklearn
# overhead on every page load.
#
# This helper builds all 160 feature rows once, performs ONE
# batch predict_proba() call, and keeps the result in memory.
# The cache is refreshed automatically every 5 minutes.
# ============================================================

LATEST_PREDICTION_CACHE_SECONDS = 300

_latest_prediction_cache = {
    "created_at": 0.0,
    "districts": None,
    "by_name": None,
}


def _json_safe_float(value, decimals):
    if pd.isna(value):
        return None

    number = float(value)

    if not np.isfinite(number):
        return None

    return round(number, decimals)


def _build_all_latest_predictions():
    """Build current next-month predictions for all districts in one batch."""

    data = (
        master_df
        .sort_values(["district", "date"])
        .copy()
    )

    # Create the same lag features used during training.
    grouped = data.groupby("district", sort=False)

    lag_sources = [
        "rainfall_mm",
        "soil_moisture",
        "ndvi",
        "temperature_c",
        "evaporation_mm",
    ]

    for feature in lag_sources:
        for lag in (1, 2, 3):
            data[f"{feature}_lag{lag}"] = (
                grouped[feature].shift(lag)
            )

    month_number = data["date"].dt.month

    data["month_sin"] = np.sin(
        2 * np.pi * month_number / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * month_number / 12
    )

    # Keep only the latest month for each district.
    latest_index = (
        data.groupby("district")["date"].idxmax()
    )

    latest = (
        data.loc[latest_index]
        .sort_values(["province", "district"])
        .reset_index(drop=True)
    )

    missing_model_columns = [
        feature
        for feature in MODEL_FEATURES
        if feature not in latest.columns
    ]

    if missing_model_columns:
        raise ValueError(
            "Unable to build batch model features: "
            + ", ".join(missing_model_columns)
        )

    feature_frame = latest[MODEL_FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid_mask = (
        feature_frame.notna().all(axis=1)
        &
        np.isfinite(
            feature_frame.to_numpy(dtype=float)
        ).all(axis=1)
    )

    valid_latest = (
        latest.loc[valid_mask]
        .reset_index(drop=True)
    )

    X = (
        feature_frame.loc[valid_mask]
        .reset_index(drop=True)
        .astype(float)
    )

    if X.empty:
        raise ValueError(
            "No valid latest district feature rows were available."
        )

    # IMPORTANT PERFORMANCE FIX:
    # one predict_proba call for all districts, not 160 calls.
    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    results = []

    for row_number, current in valid_latest.iterrows():
        probability = float(
            probabilities[row_number]
        )

        predicted_drought = int(
            probability >= WARNING_THRESHOLD
        )

        data_date = pd.Timestamp(
            current["date"]
        )

        prediction_for_date = (
            data_date
            +
            pd.offsets.MonthBegin(1)
        )

        spi_status = current.get(
            "spi_status",
            None
        )

        if pd.isna(spi_status):
            spi_status = None
        elif spi_status is not None:
            spi_status = str(spi_status)

        results.append({
            "success": True,
            "drought_probability":
                round(probability, 4),
            "drought_probability_percent":
                round(probability * 100, 2),
            "predicted_drought":
                predicted_drought,
            "prediction_status":
                (
                    "Drought Warning"
                    if predicted_drought == 1
                    else "No Drought"
                ),
            "risk_level":
                calculate_risk_level(probability),
            "warning_threshold":
                WARNING_THRESHOLD,
            "district":
                str(current["district"]),
            "province":
                str(current["province"]),
            "data_date":
                data_date.strftime("%Y-%m-%d"),
            "prediction_for_date":
                prediction_for_date.strftime("%Y-%m-%d"),
            "prediction_horizon":
                "next_month",
            "environmental_data": {
                "rainfall_mm":
                    _json_safe_float(
                        current["rainfall_mm"],
                        3
                    ),
                "rainfall_3month":
                    _json_safe_float(
                        current["rainfall_3month"],
                        3
                    ),
                "spi3":
                    _json_safe_float(
                        current["spi3"],
                        4
                    ),
                "spi_status":
                    spi_status,
                "soil_moisture":
                    _json_safe_float(
                        current["soil_moisture"],
                        4
                    ),
                "ndvi":
                    _json_safe_float(
                        current["ndvi"],
                        4
                    ),
                "temperature_c":
                    _json_safe_float(
                        current["temperature_c"],
                        2
                    ),
                "evaporation_mm":
                    _json_safe_float(
                        current["evaporation_mm"],
                        3
                    ),
            },
        })

    results.sort(
        key=lambda item:
            item["drought_probability"],
        reverse=True
    )

    by_name = {
        item["district"].strip().lower(): item
        for item in results
    }

    return results, by_name


def get_cached_latest_predictions(force=False):
    """Return cached all-district predictions, refreshing when stale."""

    now = time.monotonic()

    cache_valid = (
        not force
        and
        _latest_prediction_cache["districts"] is not None
        and
        (
            now
            -
            _latest_prediction_cache["created_at"]
        )
        <
        LATEST_PREDICTION_CACHE_SECONDS
    )

    if cache_valid:
        return (
            _latest_prediction_cache["districts"],
            _latest_prediction_cache["by_name"],
        )

    districts, by_name = (
        _build_all_latest_predictions()
    )

    _latest_prediction_cache[
        "created_at"
    ] = now

    _latest_prediction_cache[
        "districts"
    ] = districts

    _latest_prediction_cache[
        "by_name"
    ] = by_name

    return districts, by_name


def _copy_prediction(prediction):
    """Return a safe shallow/deep-enough copy for endpoint responses."""

    result = dict(prediction)

    result["environmental_data"] = dict(
        prediction.get(
            "environmental_data",
            {}
        )
    )

    return result


# ============================================================



# 14. ROOT ENDPOINT



# ============================================================







@app.get("/")



def root():







    return {



        "application": "AgriWatch",



        "status": "running",



        "model": "Drought Prediction Random Forest",



        "districts": int(



            master_df["district"].nunique()



        )



    }











# ============================================================

# 14A. API ROOT

# ============================================================



@app.get("/")

def root():

    return {

        "success": True,

        "service": "AgriWatch GeoAI API",

        "status": "online",

        "message": "AgriWatch FastAPI backend is running.",

    }



# ============================================================



# 15. HEALTH ENDPOINT



# ============================================================







@app.get("/health")



def health():







    return {



        "status": "healthy",



        "model_loaded": True,



        "dataset_loaded": True,



        "districts": int(



            master_df["district"].nunique()



        ),



        "records": int(



            len(master_df)



        )



    }











# ============================================================



# 16. MODEL INFORMATION



# ============================================================







@app.get("/model-info")



def model_info():
    return {
        "model": "AgriWatch Drought Prediction Random Forest",
        "number_of_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "warning_threshold": WARNING_THRESHOLD,
        "prediction_horizon": loaded_config.get(
            "prediction_horizon",
            "next_month"
        ),
        "target_definition": loaded_config.get(
            "target_definition",
            "next_month_spi3 <= -1"
        ),
        "available_districts": int(master_df["district"].nunique()),
        "records": int(len(master_df)),
        "latest_data_date": master_df["date"].max().strftime("%Y-%m-%d"),
    }











# ============================================================



# 17. LIST AVAILABLE DISTRICTS



# ============================================================







@app.get("/districts")



def get_districts():







    districts = (



        master_df[



            ["province", "district"]



        ]



        .drop_duplicates()



        .sort_values(



            ["province", "district"]



        )



    )







    records = districts.to_dict(



        orient="records"



    )







    return {



        "success": True,



        "count": len(records),



        "districts": records



    }











# ============================================================



# 18. ORIGINAL MANUAL PREDICTION ENDPOINT



# ============================================================







@app.post("/predict")



def predict(request: PredictionRequest):







    try:







        return run_prediction(



            request.features



        )







    except ValueError as error:







        raise HTTPException(



            status_code=400,



            detail=str(error)



        )







    except Exception as error:







        print(



            "Prediction error:",



            error



        )







        raise HTTPException(



            status_code=500,



            detail="Model prediction failed."



        )











# ============================================================



# 19. AUTOMATIC DISTRICT PREDICTION



# ============================================================







@app.get(



    "/predict-district/{district_name}"



)



def predict_district(district_name: str):
    try:
        # Resolve aliases/old names using the existing working matcher.
        district_df = find_district(district_name)

        actual_district = str(
            district_df.iloc[-1]["district"]
        )

        _, by_name = (
            get_cached_latest_predictions()
        )

        prediction = by_name.get(
            actual_district.strip().lower()
        )

        if prediction is None:
            # Very unlikely fallback; keeps endpoint behaviour safe.
            features, current = (
                build_district_features(
                    actual_district
                )
            )

            prediction = run_prediction(
                features
            )

            data_date = pd.Timestamp(
                current["date"]
            )

            prediction_for_date = (
                data_date
                +
                pd.offsets.MonthBegin(1)
            )

            spi_status = current.get(
                "spi_status",
                None
            )

            if pd.isna(spi_status):
                spi_status = None
            elif spi_status is not None:
                spi_status = str(spi_status)

            prediction.update({
                "district":
                    str(current["district"]),
                "province":
                    str(current["province"]),
                "data_date":
                    data_date.strftime("%Y-%m-%d"),
                "prediction_for_date":
                    prediction_for_date.strftime("%Y-%m-%d"),
                "prediction_horizon":
                    "next_month",
                "environmental_data": {
                    "rainfall_mm":
                        round(
                            float(current["rainfall_mm"]),
                            3
                        ),
                    "rainfall_3month":
                        round(
                            float(current["rainfall_3month"]),
                            3
                        ),
                    "spi3":
                        round(
                            float(current["spi3"]),
                            4
                        ),
                    "spi_status":
                        spi_status,
                    "soil_moisture":
                        round(
                            float(current["soil_moisture"]),
                            4
                        ),
                    "ndvi":
                        round(
                            float(current["ndvi"]),
                            4
                        ),
                    "temperature_c":
                        round(
                            float(current["temperature_c"]),
                            2
                        ),
                    "evaporation_mm":
                        round(
                            float(current["evaporation_mm"]),
                            3
                        ),
                }
            })

            return prediction

        return _copy_prediction(
            prediction
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "District prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# 20. DISTRICT HISTORICAL ML DROUGHT RISK
# Returns recent environmental history + Random Forest
# drought probability for every historical month.
# ============================================================

@app.get("/district-history/{district_name}")
def district_history(district_name: str):
    try:
        district_df = (
            find_district(district_name)
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        if district_df.empty:
            raise ValueError(
                f"No historical data available for '{district_name}'."
            )

        if len(district_df) < 4:
            raise ValueError(
                f"Not enough historical data for '{district_name}'."
            )

        actual_district = str(
            district_df.iloc[-1]["district"]
        )

        province = str(
            district_df.iloc[-1]["province"]
        )

        def safe_float(value, decimals=4):
            if pd.isna(value):
                return None

            try:
                number = float(value)

                if not np.isfinite(number):
                    return None

                return round(
                    number,
                    decimals
                )

            except (ValueError, TypeError):
                return None

        # Build all historical feature columns vectorially.
        work = district_df.copy()

        lag_sources = [
            "rainfall_mm",
            "soil_moisture",
            "ndvi",
            "temperature_c",
            "evaporation_mm",
        ]

        for feature in lag_sources:
            for lag in (1, 2, 3):
                work[
                    f"{feature}_lag{lag}"
                ] = work[feature].shift(lag)

        month_number = (
            work["date"].dt.month
        )

        work["month_sin"] = np.sin(
            2 * np.pi * month_number / 12
        )

        work["month_cos"] = np.cos(
            2 * np.pi * month_number / 12
        )

        missing_columns = [
            feature
            for feature in MODEL_FEATURES
            if feature not in work.columns
        ]

        if missing_columns:
            raise ValueError(
                "Unable to build historical model features: "
                + ", ".join(missing_columns)
            )

        X_all = work[
            MODEL_FEATURES
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        valid_mask = (
            X_all.notna().all(axis=1)
            &
            np.isfinite(
                X_all.to_numpy(dtype=float)
            ).all(axis=1)
        )

        valid_work = (
            work.loc[valid_mask]
            .copy()
            .reset_index(drop=True)
        )

        X = (
            X_all.loc[valid_mask]
            .reset_index(drop=True)
            .astype(float)
        )

        if X.empty:
            raise ValueError(
                f"Unable to create historical ML predictions "
                f"for '{district_name}'."
            )

        # One model call for the complete district history.
        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        full_history = []

        for i, current in valid_work.iterrows():
            probability = float(
                probabilities[i]
            )

            predicted_drought = int(
                probability >= WARNING_THRESHOLD
            )

            data_date = pd.Timestamp(
                current["date"]
            )

            prediction_for_date = (
                data_date
                +
                pd.offsets.MonthBegin(1)
            )

            spi_status = current.get(
                "spi_status",
                None
            )

            if pd.isna(spi_status):
                spi_status = None
            elif spi_status is not None:
                spi_status = str(spi_status)

            full_history.append({
                "date":
                    data_date.strftime("%Y-%m-%d"),
                "data_date":
                    data_date.strftime("%Y-%m-%d"),
                "prediction_for_date":
                    prediction_for_date.strftime("%Y-%m-%d"),
                "prediction_horizon":
                    "next_month",
                "year":
                    int(data_date.year),
                "month":
                    int(data_date.month),
                "month_label":
                    data_date.strftime("%b"),
                "month_year":
                    data_date.strftime("%Y-%m"),
                "rainfall_mm":
                    safe_float(
                        current["rainfall_mm"],
                        3
                    ),
                "rainfall_3month":
                    safe_float(
                        current["rainfall_3month"],
                        3
                    ),
                "spi3":
                    safe_float(
                        current["spi3"],
                        4
                    ),
                "spi_status":
                    spi_status,
                "soil_moisture":
                    safe_float(
                        current["soil_moisture"],
                        4
                    ),
                "ndvi":
                    safe_float(
                        current["ndvi"],
                        4
                    ),
                "temperature_c":
                    safe_float(
                        current["temperature_c"],
                        2
                    ),
                "evaporation_mm":
                    safe_float(
                        current["evaporation_mm"],
                        3
                    ),
                "drought_probability":
                    round(probability, 4),
                "drought_probability_percent":
                    round(
                        probability * 100,
                        2
                    ),
                "predicted_drought":
                    predicted_drought,
                "prediction_status":
                    (
                        "Drought Warning"
                        if predicted_drought == 1
                        else "No Drought"
                    ),
                "risk_level":
                    calculate_risk_level(
                        probability
                    ),
            })

        history = full_history[-24:]
        latest = history[-1]

        return {
            "success": True,
            "district": actual_district,
            "province": province,
            "records": len(history),
            "latest_date": latest["date"],
            "latest_data_date":
                latest["data_date"],
            "latest_prediction_for_date":
                latest["prediction_for_date"],
            "prediction_horizon":
                "next_month",
            "warning_threshold":
                WARNING_THRESHOLD,
            "current": {
                "rainfall_mm":
                    latest["rainfall_mm"],
                "rainfall_3month":
                    latest["rainfall_3month"],
                "spi3":
                    latest["spi3"],
                "spi_status":
                    latest["spi_status"],
                "soil_moisture":
                    latest["soil_moisture"],
                "ndvi":
                    latest["ndvi"],
                "temperature_c":
                    latest["temperature_c"],
                "evaporation_mm":
                    latest["evaporation_mm"],
                "drought_probability":
                    latest["drought_probability"],
                "drought_probability_percent":
                    latest["drought_probability_percent"],
                "predicted_drought":
                    latest["predicted_drought"],
                "prediction_status":
                    latest["prediction_status"],
                "risk_level":
                    latest["risk_level"],
            },
            "history":
                history,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "DISTRICT HISTORY ML ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )




# ============================================================



# 21. COMPARE ALL DISTRICTS



# ============================================================







@app.get("/compare-districts")



def compare_districts():
    try:
        predictions, _ = (
            get_cached_latest_predictions()
        )

        comparison_results = []

        for prediction in predictions:
            comparison_results.append({
                "district":
                    prediction["district"],
                "province":
                    prediction["province"],
                "drought_probability":
                    prediction["drought_probability"],
                "drought_probability_percent":
                    prediction[
                        "drought_probability_percent"
                    ],
                "predicted_drought":
                    prediction["predicted_drought"],
                "prediction_status":
                    prediction["prediction_status"],
                "risk_level":
                    prediction["risk_level"],
                "data_date":
                    prediction.get("data_date"),
                "prediction_for_date":
                    prediction.get(
                        "prediction_for_date"
                    ),
                "prediction_horizon":
                    prediction.get(
                        "prediction_horizon",
                        "next_month"
                    ),
                "environmental_data":
                    dict(
                        prediction.get(
                            "environmental_data",
                            {}
                        )
                    ),
            })

        return {
            "success": True,
            "count":
                len(comparison_results),
            "warning_threshold":
                WARNING_THRESHOLD,
            "districts":
                comparison_results,
        }

    except Exception as error:
        print(
            "COMPARE DISTRICTS ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )




# ============================================================

# AGRIWATCH MAP DATA



# Returns latest ML drought data + coordinates for all districts



# ============================================================







@app.get("/map-data")



def get_map_data():
    """Return cached live predictions + coordinates for all districts."""

    try:
        predictions, _ = (
            get_cached_latest_predictions()
        )

        coordinates_path = os.path.join(
            BASE_DIR,
            "data",
            "district_coordinates.csv"
        )

        coords_lookup = {}

        def normalize_coord_name(value):
            value = str(
                value or ""
            ).strip().lower()

            value = re.sub(
                r"\b(district|agency)\b",
                "",
                value
            )

            value = re.sub(
                r"[^a-z0-9]+",
                "",
                value
            )

            return value

        if os.path.exists(
            coordinates_path
        ):
            coords_df = pd.read_csv(
                coordinates_path
            )

            required_coord_columns = {
                "district",
                "latitude",
                "longitude"
            }

            if required_coord_columns.issubset(
                coords_df.columns
            ):
                for _, coord_row in (
                    coords_df.iterrows()
                ):
                    district_key = (
                        normalize_coord_name(
                            coord_row.get(
                                "district"
                            )
                        )
                    )

                    province_key = str(
                        coord_row.get(
                            "province",
                            ""
                        )
                    ).strip().lower()

                    latitude = pd.to_numeric(
                        coord_row.get(
                            "latitude"
                        ),
                        errors="coerce"
                    )

                    longitude = pd.to_numeric(
                        coord_row.get(
                            "longitude"
                        ),
                        errors="coerce"
                    )

                    if (
                        district_key
                        and
                        pd.notna(latitude)
                        and
                        pd.notna(longitude)
                    ):
                        coord = (
                            float(latitude),
                            float(longitude)
                        )

                        coords_lookup[
                            (
                                province_key,
                                district_key
                            )
                        ] = coord

                        coords_lookup[
                            (
                                "",
                                district_key
                            )
                        ] = coord

        districts = []
        coordinates_found = 0

        for prediction in predictions:
            actual_district = (
                prediction["district"]
            )

            province = (
                prediction["province"]
            )

            key = normalize_coord_name(
                actual_district
            )

            coord = coords_lookup.get(
                (
                    province.strip().lower(),
                    key
                )
            )

            if coord is None:
                coord = coords_lookup.get(
                    ("", key)
                )

            latitude = None
            longitude = None

            if coord is not None:
                latitude = round(
                    float(coord[0]),
                    6
                )

                longitude = round(
                    float(coord[1]),
                    6
                )

                coordinates_found += 1

            item = _copy_prediction(
                prediction
            )

            item["latitude"] = latitude
            item["longitude"] = longitude

            districts.append(
                item
            )

        return {
            "success": True,
            "count":
                len(districts),
            "total_risk_records":
                len(districts),
            "available_districts":
                int(
                    master_df[
                        "district"
                    ].nunique()
                ),
            "warning_threshold":
                WARNING_THRESHOLD,
            "coordinates_found":
                coordinates_found,
            "coordinates_missing":
                len(districts)
                -
                coordinates_found,
            "prediction_errors":
                [],
            "cache_seconds":
                LATEST_PREDICTION_CACHE_SECONDS,
            "districts":
                districts,
        }

    except Exception as error:
        print(
            "MAP DATA ERROR:",
            error
        )

        return {
            "success": False,
            "error":
                str(error),
            "count":
                0,
            "districts":
                []
        }




# ============================================================

# 23. NATIONAL HISTORICAL ENVIRONMENTAL TREND



# Returns monthly Pakistan-wide average NDVI and soil moisture



# ============================================================







@app.get("/national-history")



def get_national_history():







    try:



        # ----------------------------------------------------



        # COPY ONLY THE COLUMNS WE NEED



        # ----------------------------------------------------







        history_df = master_df[



            [



                "date",



                "district",



                "province",



                "ndvi",



                "soil_moisture"



            ]



        ].copy()







        # ----------------------------------------------------



        # REMOVE INVALID RECORDS



        # ----------------------------------------------------







        history_df = history_df.dropna(



            subset=[



                "date",



                "ndvi",



                "soil_moisture"



            ]



        )







        if history_df.empty:



            raise ValueError(



                "No historical environmental data available."



            )







        # ----------------------------------------------------



        # CREATE MONTH COLUMN



        # ----------------------------------------------------







        history_df["month"] = (



            history_df["date"]



            .dt.to_period("M")



            .astype(str)



        )







        # ----------------------------------------------------



        # NATIONAL MONTHLY AVERAGES



        #



        # Every district contributes to the Pakistan-wide



        # average for each month.



        # ----------------------------------------------------







        monthly = (



            history_df



            .groupby("month", as_index=False)



            .agg(



                ndvi=("ndvi", "mean"),



                soil_moisture=("soil_moisture", "mean"),



                districts=("district", "nunique")



            )



            .sort_values("month")



            .reset_index(drop=True)



        )







        # ----------------------------------------------------



        # WEBSITE ONLY NEEDS RECENT 12 MONTHS



        # ----------------------------------------------------







        monthly = monthly.tail(12)







        history = []







        for _, row in monthly.iterrows():







            history.append({



                "month": str(row["month"]),







                "ndvi": round(



                    float(row["ndvi"]),



                    4



                ),







                "soil_moisture": round(



                    float(row["soil_moisture"]),



                    4



                ),







                "districts": int(



                    row["districts"]



                )



            })







        # ----------------------------------------------------



        # CURRENT NATIONAL VALUES



        # ----------------------------------------------------







        latest = history[-1]







        # ----------------------------------------------------



        # RESPONSE



        # ----------------------------------------------------







        return {



            "success": True,







            "records": len(history),







            "total_districts": int(



                master_df["district"].nunique()



            ),







            "latest_month": latest["month"],







            "current_ndvi": latest["ndvi"],







            "current_soil_moisture":



                latest["soil_moisture"],







            "history": history



        }







    except Exception as error:







        print(



            "NATIONAL HISTORY ERROR:",



            error



        )







        raise HTTPException(



            status_code=500,



            detail=str(error)



        )







# ============================================================



# 24. AGRIWATCH ALERT MANAGEMENT



# Stores dispatched alerts in alerts.json



# ============================================================











# ------------------------------------------------------------



# ALERT STORAGE PATH



# ------------------------------------------------------------







ALERTS_FILE = os.path.join(



    BASE_DIR,



    "alerts.json"



)











# ------------------------------------------------------------



# ALERT REQUEST MODEL



# ------------------------------------------------------------







class AlertRequest(BaseModel):



    district: str



    severity: str



    message: str



    audience: str











# ------------------------------------------------------------



# LOAD ALERTS FROM JSON



# ------------------------------------------------------------







def load_alerts():







    # If file does not exist yet,



    # simply return an empty alert list.



    if not os.path.exists(ALERTS_FILE):



        return []







    try:







        with open(



            ALERTS_FILE,



            "r",



            encoding="utf-8"



        ) as file:







            data = json.load(file)







        if isinstance(data, list):



            return data







        return []







    except Exception as error:







        print(



            "ALERT FILE LOAD ERROR:",



            error



        )







        return []











# ------------------------------------------------------------



# SAVE ALERTS TO JSON



# ------------------------------------------------------------







def save_alerts(alerts):







    try:







        with open(



            ALERTS_FILE,



            "w",



            encoding="utf-8"



        ) as file:







            json.dump(



                alerts,



                file,



                indent=2,



                ensure_ascii=False



            )







    except Exception as error:







        print(



            "ALERT FILE SAVE ERROR:",



            error



        )







        raise











# ============================================================



# GET ALERT HISTORY



# ============================================================







@app.get("/alerts")



def get_alerts():







    try:







        alerts = load_alerts()







        return {



            "success": True,



            "count": len(alerts),



            "alerts": alerts



        }







    except Exception as error:







        print(



            "GET ALERTS ERROR:",



            error



        )







        raise HTTPException(



            status_code=500,



            detail="Failed to load alert history."



        )











# ============================================================



# CREATE / DISPATCH ALERT



# ============================================================







@app.post("/alerts")



def create_alert(



    alert: AlertRequest



):







    try:







        # ----------------------------------------------------



        # CLEAN INPUT



        # ----------------------------------------------------







        district = alert.district.strip()







        severity = alert.severity.strip()







        message = alert.message.strip()







        audience = alert.audience.strip()











        # ----------------------------------------------------



        # VALIDATION



        # ----------------------------------------------------







        if not district:







            raise HTTPException(



                status_code=400,



                detail="District is required."



            )











        if not message:







            raise HTTPException(



                status_code=400,



                detail="Alert message is required."



            )











        allowed_severities = [



            "Normal",



            "Moderate",



            "Severe",



            "Extreme"



        ]











        if severity not in allowed_severities:







            raise HTTPException(



                status_code=400,



                detail=(



                    "Invalid severity. "



                    "Allowed values: "



                    + ", ".join(



                        allowed_severities



                    )



                )



            )











        allowed_audiences = [



            "Farmers + Public",



            "Farmers",



            "General Public"



        ]











        if audience not in allowed_audiences:







            raise HTTPException(



                status_code=400,



                detail=(



                    "Invalid audience. "



                    "Allowed values: "



                    + ", ".join(



                        allowed_audiences



                    )



                )



            )











        # ----------------------------------------------------



        # VERIFY DISTRICT EXISTS



        # ----------------------------------------------------







        district_matches = master_df[



            master_df["district"]



            .str.lower()



            == district.lower()



        ]











        if district_matches.empty:







            raise HTTPException(



                status_code=404,



                detail=(



                    f"District '{district}' "



                    "was not found."



                )



            )











        # Use exact official district name



        actual_district = str(



            district_matches.iloc[0][



                "district"



            ]



        )











        province = str(



            district_matches.iloc[0][



                "province"



            ]



        )











        # ----------------------------------------------------



        # LOAD EXISTING ALERTS



        # ----------------------------------------------------







        alerts = load_alerts()











        # ----------------------------------------------------



        # GENERATE NEXT ALERT ID



        # ----------------------------------------------------







        existing_numbers = []











        for existing_alert in alerts:







            try:







                alert_id = str(



                    existing_alert.get(



                        "id",



                        ""



                    )



                )











                number = int(



                    alert_id.replace(



                        "AL-",



                        ""



                    )



                )











                existing_numbers.append(



                    number



                )







            except (



                ValueError,



                TypeError



            ):







                continue











        if existing_numbers:







            next_number = (



                max(existing_numbers)



                + 1



            )







        else:







            next_number = 1001











        alert_id = (



            f"AL-{next_number}"



        )











        # ----------------------------------------------------



        # DATE / TIME



        # ----------------------------------------------------







        from datetime import datetime











        current_time = datetime.now()











        # ----------------------------------------------------



        # CREATE ALERT RECORD



        # ----------------------------------------------------







        new_alert = {







            "id":



                alert_id,







            "district":



                actual_district,







            "province":



                province,







            "severity":



                severity,







            "message":



                message,







            "sentTo":



                audience,







            "date":



                current_time.strftime(



                    "%Y-%m-%d"



                ),







            "timestamp":



                current_time.isoformat(),







            "status":



                "Delivered"



        }











        # ----------------------------------------------------



        # NEWEST ALERT FIRST



        # ----------------------------------------------------







        alerts.insert(



            0,



            new_alert



        )











        # ----------------------------------------------------



        # SAVE



        # ----------------------------------------------------







        save_alerts(



            alerts



        )











        print(



            "=" * 60



        )







        print(



            "AGRIWATCH ALERT DISPATCHED"



        )







        print(



            "=" * 60



        )







        print(



            "ID:",



            alert_id



        )







        print(



            "District:",



            actual_district



        )







        print(



            "Province:",



            province



        )







        print(



            "Severity:",



            severity



        )







        print(



            "Audience:",



            audience



        )







        print(



            "=" * 60



        )











        # ----------------------------------------------------



        # RESPONSE



        # ----------------------------------------------------







        return {







            "success": True,







            "message":



                "Alert dispatched successfully.",







            "alert":



                new_alert



        }











    # --------------------------------------------------------



    # KEEP HTTP ERRORS



    # --------------------------------------------------------







    except HTTPException:







        raise











    # --------------------------------------------------------



    # SERVER ERROR



    # --------------------------------------------------------







    except Exception as error:







        print(



            "CREATE ALERT ERROR:",



            error



        )







        raise HTTPException(



            status_code=500,



            detail=(



                "Failed to dispatch alert."



            )



        )

# ============================================================
# 25. IRRIGATION CONTEXT FOR FARMER PORTAL
# Returns real district ML/environmental data + map coordinates.
# The frontend combines this with the 7-day weather forecast.
# ============================================================

@app.get("/irrigation-context/{district_name}")
def irrigation_context(district_name: str):
    try:
        prediction = predict_district(district_name)
        actual_district = prediction["district"]
        province = prediction["province"]

        coordinates_path = os.path.join(
            BASE_DIR, "data", "district_coordinates.csv"
        )

        if not os.path.exists(coordinates_path):
            raise HTTPException(
                status_code=500,
                detail="district_coordinates.csv was not found."
            )

        coords_df = pd.read_csv(coordinates_path)

        def normalize_coord_name(value):
            value = str(value or "").strip().lower()
            value = re.sub(r"\b(district|agency)\b", "", value)
            value = re.sub(r"[^a-z0-9]+", "", value)
            return value

        coords_df["_district_key"] = coords_df["district"].map(
            normalize_coord_name
        )

        district_key = normalize_coord_name(actual_district)

        # Prefer same-province match when province exists, then fall back to
        # district name only so old province labels do not break the endpoint.
        if "province" in coords_df.columns:
            coords_df["_province_key"] = (
                coords_df["province"].astype(str).str.strip().str.lower()
            )
            match = coords_df[
                (coords_df["_district_key"] == district_key)
                & (coords_df["_province_key"] == province.strip().lower())
            ]
        else:
            match = coords_df.iloc[0:0]

        if match.empty:
            match = coords_df[
                coords_df["_district_key"] == district_key
            ]

        if match.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Coordinates not found for '{actual_district}'. "
                    "The ML prediction is available, but this new district "
                    "still needs a row in district_coordinates.csv for live weather."
                )
            )

        row = match.iloc[0]
        env = prediction.get("environmental_data", {})

        return {
            "success": True,
            "district": actual_district,
            "province": province,
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
            "data_date": prediction.get("data_date"),
            "prediction_for_date": prediction.get("prediction_for_date"),
            "prediction_horizon": prediction.get("prediction_horizon", "next_month"),
            "drought_probability": prediction.get("drought_probability"),
            "drought_probability_percent": prediction.get("drought_probability_percent"),
            "predicted_drought": prediction.get("predicted_drought"),
            "prediction_status": prediction.get("prediction_status"),
            "risk_level": prediction.get("risk_level"),
            "warning_threshold": prediction.get("warning_threshold"),
            "environmental_data": {
                "rainfall_mm": env.get("rainfall_mm"),
                "rainfall_3month": env.get("rainfall_3month"),
                "spi3": env.get("spi3"),
                "spi_status": env.get("spi_status"),
                "soil_moisture": env.get("soil_moisture"),
                "ndvi": env.get("ndvi"),
                "temperature_c": env.get("temperature_c"),
                "evaporation_mm": env.get("evaporation_mm"),
            },
        }

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        print("IRRIGATION CONTEXT ERROR:", error)
        raise HTTPException(status_code=500, detail=str(error))
# ============================================================
# 26. IRRIGATION RECOMMENDATION
# ============================================================
# Calculates irrigation guidance for a farmer field using:
# - crop
# - field area
# - district environmental data
# - drought ML prediction
# ============================================================


class IrrigationRecommendationRequest(BaseModel):
    district: str
    field_id: str | None = None
    field_name: str = "Field"
    crop: str
    area_acres: float


# ============================================================
# CROP IRRIGATION CONFIGURATION
# ============================================================

CROP_IRRIGATION_CONFIG = {

    "wheat": {
        "base_water_mm": 35.0,
        "soil_threshold": 0.30,
    },

    "cotton": {
        "base_water_mm": 45.0,
        "soil_threshold": 0.35,
    },

    "rice": {
        "base_water_mm": 60.0,
        "soil_threshold": 0.40,
    },

    "maize": {
        "base_water_mm": 40.0,
        "soil_threshold": 0.35,
    },

    "corn": {
        "base_water_mm": 40.0,
        "soil_threshold": 0.35,
    },

    "sugarcane": {
        "base_water_mm": 55.0,
        "soil_threshold": 0.40,
    },

    "potato": {
        "base_water_mm": 35.0,
        "soil_threshold": 0.35,
    },

    "tomato": {
        "base_water_mm": 30.0,
        "soil_threshold": 0.35,
    },

    "onion": {
        "base_water_mm": 25.0,
        "soil_threshold": 0.30,
    },

    "vegetables": {
        "base_water_mm": 30.0,
        "soil_threshold": 0.35,
    },
}


# ============================================================
# GET CROP CONFIGURATION
# ============================================================

def get_crop_irrigation_config(crop_name: str):

    crop = str(crop_name or "").strip().lower()

    # Direct crop match
    if crop in CROP_IRRIGATION_CONFIG:
        return CROP_IRRIGATION_CONFIG[crop]

    # Handle common naming variations
    aliases = {
        "maize/corn": "maize",
        "corn/maize": "maize",
        "paddy": "rice",
        "paddy rice": "rice",
        "vegetable": "vegetables",
    }

    if crop in aliases:
        return CROP_IRRIGATION_CONFIG[aliases[crop]]

    # Safe default for unknown crops
    return {
        "base_water_mm": 35.0,
        "soil_threshold": 0.35,
    }


# ============================================================
# POST IRRIGATION RECOMMENDATION
# ============================================================

@app.post("/irrigation-recommendation")
def irrigation_recommendation(
    request: IrrigationRecommendationRequest
):

    try:

        # ----------------------------------------------------
        # VALIDATE REQUEST
        # ----------------------------------------------------

        district = str(request.district or "").strip()
        crop = str(request.crop or "").strip()
        field_name = str(
            request.field_name or "Field"
        ).strip()

        area_acres = float(request.area_acres)

        if not district:
            raise HTTPException(
                status_code=400,
                detail="District is required."
            )

        if not crop:
            raise HTTPException(
                status_code=400,
                detail="Crop is required."
            )

        if area_acres <= 0:
            raise HTTPException(
                status_code=400,
                detail="Field area must be greater than 0 acres."
            )


        print("=" * 60)
        print("IRRIGATION RECOMMENDATION REQUEST")
        print("=" * 60)

        print("District:", district)
        print("Field ID:", request.field_id)
        print("Field:", field_name)
        print("Crop:", crop)
        print("Area:", area_acres)

        print("=" * 60)


        # ----------------------------------------------------
        # GET REAL DISTRICT ML + ENVIRONMENTAL DATA
        # ----------------------------------------------------

        features, current = build_district_features(
            district
        )

        prediction = run_prediction(
            features
        )


        # ----------------------------------------------------
        # REAL ENVIRONMENTAL VALUES
        # ----------------------------------------------------

        actual_district = str(
            current["district"]
        )

        province = str(
            current["province"]
        )

        data_date = current["date"].strftime(
            "%Y-%m-%d"
        )


        rainfall_mm = float(
            current["rainfall_mm"]
        )

        soil_moisture = float(
            current["soil_moisture"]
        )

        ndvi = float(
            current["ndvi"]
        )

        temperature_c = float(
            current["temperature_c"]
        )

        evaporation_mm = float(
            current["evaporation_mm"]
        )


        # ----------------------------------------------------
        # ML VALUES
        # ----------------------------------------------------

        drought_probability = float(
            prediction.get(
                "drought_probability",
                0
            )
        )

        drought_probability_percent = float(
            prediction.get(
                "drought_probability_percent",
                drought_probability * 100
            )
        )

        risk_level = str(
            prediction.get(
                "risk_level",
                "Low"
            )
        )

        predicted_drought = int(
            prediction.get(
                "predicted_drought",
                0
            )
        )


        # ----------------------------------------------------
        # CROP WATER REQUIREMENT
        # ----------------------------------------------------

        crop_config = get_crop_irrigation_config(
            crop
        )

        base_water_mm = float(
            crop_config["base_water_mm"]
        )

        soil_threshold = float(
            crop_config["soil_threshold"]
        )


        # ----------------------------------------------------
        # SOIL MOISTURE FACTOR
        # ----------------------------------------------------
        #
        # Lower soil moisture = more irrigation needed.
        # ----------------------------------------------------

        if soil_moisture <= 0.15:

            moisture_factor = 1.30

            moisture_status = "Very Dry"

        elif soil_moisture <= 0.25:

            moisture_factor = 1.15

            moisture_status = "Dry"

        elif soil_moisture < soil_threshold:

            moisture_factor = 1.00

            moisture_status = "Moderate"

        elif soil_moisture < 0.45:

            moisture_factor = 0.65

            moisture_status = "Adequate"

        else:

            moisture_factor = 0.25

            moisture_status = "Wet"


        # ----------------------------------------------------
        # DROUGHT FACTOR
        # ----------------------------------------------------

        if drought_probability >= 0.65:

            drought_factor = 1.25

        elif drought_probability >= 0.50:

            drought_factor = 1.15

        elif drought_probability >= 0.25:

            drought_factor = 1.05

        else:

            drought_factor = 1.00


        # ----------------------------------------------------
        # TEMPERATURE FACTOR
        # ----------------------------------------------------

        if temperature_c >= 40:

            temperature_factor = 1.20

        elif temperature_c >= 35:

            temperature_factor = 1.10

        elif temperature_c >= 30:

            temperature_factor = 1.05

        else:

            temperature_factor = 1.00


        # ----------------------------------------------------
        # VEGETATION / NDVI FACTOR
        # ----------------------------------------------------

        if ndvi < 0.20:

            ndvi_factor = 1.10

        elif ndvi < 0.35:

            ndvi_factor = 1.05

        else:

            ndvi_factor = 1.00


        # ----------------------------------------------------
        # RAW IRRIGATION REQUIREMENT
        # ----------------------------------------------------

        calculated_water_mm = (
            base_water_mm
            * moisture_factor
            * drought_factor
            * temperature_factor
            * ndvi_factor
        )


        # ----------------------------------------------------
        # RAINFALL ADJUSTMENT
        # ----------------------------------------------------
        #
        # Current dataset rainfall is monthly rainfall.
        # We should NOT subtract the whole monthly rainfall
        # directly from one irrigation event.
        #
        # Instead it influences the recommendation level.
        # ----------------------------------------------------

        if rainfall_mm >= 200:

            rainfall_factor = 0.25

        elif rainfall_mm >= 120:

            rainfall_factor = 0.45

        elif rainfall_mm >= 70:

            rainfall_factor = 0.70

        elif rainfall_mm >= 30:

            rainfall_factor = 0.85

        else:

            rainfall_factor = 1.00


        calculated_water_mm *= rainfall_factor


        # ----------------------------------------------------
        # EVAPORATION ADJUSTMENT
        # ----------------------------------------------------

        if evaporation_mm >= 8:

            calculated_water_mm *= 1.15

        elif evaporation_mm >= 5:

            calculated_water_mm *= 1.08


        # ----------------------------------------------------
        # DETERMINE WHETHER IRRIGATION IS REQUIRED
        # ----------------------------------------------------

        irrigation_required = True


        # Very wet conditions
        if (
            soil_moisture >= 0.45
            and rainfall_mm >= 70
        ):

            irrigation_required = False


        # Strong rainfall + reasonable soil moisture
        elif (
            rainfall_mm >= 150
            and soil_moisture >= 0.25
            and drought_probability < 0.50
        ):

            irrigation_required = False


        # Tiny calculated amount is not worth irrigation
        elif calculated_water_mm < 5:

            irrigation_required = False


        # ----------------------------------------------------
        # FINAL WATER DEPTH
        # ----------------------------------------------------

        if irrigation_required:

            recommended_water_mm = round(
                max(
                    5.0,
                    calculated_water_mm
                ),
                1
            )

        else:

            recommended_water_mm = 0.0


        # ----------------------------------------------------
        # CONVERT MM -> LITERS
        # ----------------------------------------------------
        #
        # 1 acre = 4046.8564224 m²
        #
        # 1 mm water over 1 m² = 1 liter
        # ----------------------------------------------------

        area_square_meters = (
            area_acres
            * 4046.8564224
        )


        total_water_liters = (
            recommended_water_mm
            * area_square_meters
        )


        total_water_m3 = (
            total_water_liters
            / 1000
        )


        # ----------------------------------------------------
        # IRRIGATION PRIORITY
        # ----------------------------------------------------

        if not irrigation_required:

            priority = "Low"

        elif (
            soil_moisture < 0.15
            or drought_probability >= 0.65
        ):

            priority = "Critical"

        elif (
            soil_moisture < 0.25
            or drought_probability >= 0.50
        ):

            priority = "High"

        elif (
            soil_moisture < soil_threshold
        ):

            priority = "Medium"

        else:

            priority = "Low"


        # ----------------------------------------------------
        # RECOMMENDED TIMING
        # ----------------------------------------------------

        if not irrigation_required:

            timing = (
                "No immediate irrigation required"
            )

        elif priority == "Critical":

            timing = (
                "Irrigate as soon as possible, "
                "preferably early morning or evening"
            )

        elif priority == "High":

            timing = (
                "Irrigate within the next 24 hours, "
                "preferably early morning"
            )

        elif priority == "Medium":

            timing = (
                "Irrigate within the next 1–2 days"
            )

        else:

            timing = (
                "Monitor soil moisture before irrigation"
            )


        # ----------------------------------------------------
        # CREATE HUMAN-READABLE RECOMMENDATION
        # ----------------------------------------------------

        if not irrigation_required:

            recommendation = (
                f"No immediate irrigation is recommended "
                f"for {field_name}. "
                f"Current soil moisture is "
                f"{soil_moisture * 100:.1f}% and "
                f"monthly rainfall is "
                f"{rainfall_mm:.1f} mm. "
                f"Continue monitoring field conditions."
            )

        else:

            recommendation = (
                f"Irrigation is recommended for "
                f"{field_name}. Apply approximately "
                f"{recommended_water_mm:.1f} mm of water "
                f"to the {crop} crop. "
                f"Current soil moisture is "
                f"{soil_moisture * 100:.1f}% and "
                f"drought probability is "
                f"{drought_probability_percent:.1f}%."
            )


        # ----------------------------------------------------
        # REASONS
        # ----------------------------------------------------

        reasons = []


        if soil_moisture < soil_threshold:

            reasons.append(
                f"Soil moisture "
                f"({soil_moisture * 100:.1f}%) "
                f"is below the preferred level "
                f"for {crop}."
            )

        else:

            reasons.append(
                f"Soil moisture is currently "
                f"{soil_moisture * 100:.1f}%."
            )


        if rainfall_mm >= 150:

            reasons.append(
                f"Monthly rainfall is high "
                f"({rainfall_mm:.1f} mm), "
                f"so irrigation demand has been reduced."
            )

        elif rainfall_mm < 30:

            reasons.append(
                f"Monthly rainfall is low "
                f"({rainfall_mm:.1f} mm)."
            )


        if drought_probability >= 0.50:

            reasons.append(
                f"The drought model reports "
                f"{drought_probability_percent:.1f}% "
                f"drought probability."
            )


        if temperature_c >= 35:

            reasons.append(
                f"High temperature "
                f"({temperature_c:.1f}°C) "
                f"increases crop water demand."
            )


        if ndvi < 0.30:

            reasons.append(
                f"NDVI is relatively low "
                f"({ndvi:.3f}), indicating vegetation stress."
            )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "field": {

                "field_id": request.field_id,

                "field_name": field_name,

                "crop": crop,

                "area_acres": round(
                    area_acres,
                    2
                ),

            },


            "district": actual_district,

            "province": province,

            "data_date": data_date,


            "irrigation_required":
                irrigation_required,


            "priority":
                priority,


            "recommended_water_mm":
                recommended_water_mm,


            "recommended_water_liters":
                round(
                    total_water_liters,
                    2
                ),


            "recommended_water_m3":
                round(
                    total_water_m3,
                    2
                ),


            "recommended_timing":
                timing,


            "recommendation":
                recommendation,


            "reasons":
                reasons,


            "environmental_data": {

                "soil_moisture":
                    round(
                        soil_moisture,
                        4
                    ),

                "soil_moisture_percent":
                    round(
                        soil_moisture * 100,
                        1
                    ),

                "rainfall_mm":
                    round(
                        rainfall_mm,
                        2
                    ),

                "ndvi":
                    round(
                        ndvi,
                        4
                    ),

                "temperature_c":
                    round(
                        temperature_c,
                        2
                    ),

                "evaporation_mm":
                    round(
                        evaporation_mm,
                        3
                    ),

            },


            "drought": {

                "probability":
                    round(
                        drought_probability,
                        4
                    ),

                "probability_percent":
                    round(
                        drought_probability_percent,
                        2
                    ),

                "risk_level":
                    risk_level,

                "predicted_drought":
                    predicted_drought,

            },

        }


        print("IRRIGATION RESULT:")
        print(
            "Required:",
            irrigation_required
        )

        print(
            "Priority:",
            priority
        )

        print(
            "Water:",
            recommended_water_mm,
            "mm"
        )

        print(
            "Water volume:",
            round(total_water_m3, 2),
            "m3"
        )

        print("=" * 60)


        return response


    # --------------------------------------------------------
    # PRESERVE FASTAPI ERRORS
    # --------------------------------------------------------

    except HTTPException:

        raise


    # --------------------------------------------------------
    # DISTRICT / VALIDATION ERRORS
    # --------------------------------------------------------

    except ValueError as error:

        print(
            "IRRIGATION VALUE ERROR:",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    # --------------------------------------------------------
    # UNEXPECTED SERVER ERROR
    # --------------------------------------------------------

    except Exception as error:

        print(
            "IRRIGATION RECOMMENDATION ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to calculate irrigation "
                f"recommendation: {str(error)}"
            )
        )
