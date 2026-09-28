# ============================================================



# AGRIWATCH BACKEND API



# Drought Prediction System



# ============================================================







import os



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







    requested = (



        district_name



        .strip()



        .lower()



    )







    # --------------------------------------------------------



    # Exact match first



    # --------------------------------------------------------







    exact = master_df[



        master_df["district"]



        .str.lower()



        == requested



    ]







    if not exact.empty:



        return exact.copy()







    # --------------------------------------------------------



    # Try adding "District"



    # Example:



    # Jacobabad -> Jacobabad District



    # --------------------------------------------------------







    if not requested.endswith(" district"):







        requested_with_district = (



            requested + " district"



        )







        exact_with_district = master_df[



            master_df["district"]



            .str.lower()



            == requested_with_district



        ]







        if not exact_with_district.empty:



            return exact_with_district.copy()







    # --------------------------------------------------------



    # Partial match



    # --------------------------------------------------------







    partial = master_df[



        master_df["district"]



        .str.lower()



        .str.contains(



            requested,



            regex=False,



            na=False



        )



    ]







    unique_matches = (



        partial["district"]



        .drop_duplicates()



        .tolist()



    )







    if len(unique_matches) == 1:







        return master_df[



            master_df["district"]



            == unique_matches[0]



        ].copy()







    if len(unique_matches) > 1:







        raise ValueError(



            "Multiple districts matched. Matches: "



            + ", ".join(unique_matches)



        )







    raise ValueError(



        f"District '{district_name}' was not found."



    )











# ============================================================



# 13. BUILD 22 FEATURES AUTOMATICALLY



# ============================================================







def build_district_features(



    district_name: str



):







    district_df = find_district(



        district_name



    )







    district_df = (



        district_df



        .sort_values("date")



        .reset_index(drop=True)



    )







    # Current month + previous 3 months required



    if len(district_df) < 4:







        raise ValueError(



            f"Not enough historical data for "



            f"'{district_name}'."



        )







    current = district_df.iloc[-1]



    lag1 = district_df.iloc[-2]



    lag2 = district_df.iloc[-3]



    lag3 = district_df.iloc[-4]







    month = int(



        current["date"].month



    )







    month_sin = np.sin(



        2 * np.pi * month / 12



    )







    month_cos = np.cos(



        2 * np.pi * month / 12



    )







    features = {







        # Current environmental conditions



        "rainfall_mm":



            float(current["rainfall_mm"]),







        "soil_moisture":



            float(current["soil_moisture"]),







        "ndvi":



            float(current["ndvi"]),







        "temperature_c":



            float(current["temperature_c"]),







        "evaporation_mm":



            float(current["evaporation_mm"]),











        # Rainfall lags



        "rainfall_mm_lag1":



            float(lag1["rainfall_mm"]),







        "rainfall_mm_lag2":



            float(lag2["rainfall_mm"]),







        "rainfall_mm_lag3":



            float(lag3["rainfall_mm"]),











        # Soil moisture lags



        "soil_moisture_lag1":



            float(lag1["soil_moisture"]),







        "soil_moisture_lag2":



            float(lag2["soil_moisture"]),







        "soil_moisture_lag3":



            float(lag3["soil_moisture"]),











        # NDVI lags



        "ndvi_lag1":



            float(lag1["ndvi"]),







        "ndvi_lag2":



            float(lag2["ndvi"]),







        "ndvi_lag3":



            float(lag3["ndvi"]),











        # Temperature lags



        "temperature_c_lag1":



            float(lag1["temperature_c"]),







        "temperature_c_lag2":



            float(lag2["temperature_c"]),







        "temperature_c_lag3":



            float(lag3["temperature_c"]),











        # Evaporation lags



        "evaporation_mm_lag1":



            float(lag1["evaporation_mm"]),







        "evaporation_mm_lag2":



            float(lag2["evaporation_mm"]),







        "evaporation_mm_lag3":



            float(lag3["evaporation_mm"]),











        # Seasonal encoding



        "month_sin":



            float(month_sin),







        "month_cos":



            float(month_cos),



    }







    # Make sure no NaN values reach model



    invalid_features = [



        key



        for key, value in features.items()



        if pd.isna(value)



    ]







    if invalid_features:







        raise ValueError(



            "Missing environmental values for "



            f"{district_name}: "



            + ", ".join(invalid_features)



        )







    return features, current











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



        "model":



            "AgriWatch Drought Prediction Random Forest",







        "number_of_features":



            len(MODEL_FEATURES),







        "features":



            MODEL_FEATURES,







        "warning_threshold":



            WARNING_THRESHOLD,







        "available_districts":



            int(



                master_df["district"].nunique()



            )



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



def predict_district(



    district_name: str



):







    try:







        features, current = (



            build_district_features(



                district_name



            )



        )







        prediction = run_prediction(



            features



        )







        prediction.update({







            "district":



                str(current["district"]),







            "province":



                str(current["province"]),







            "data_date":



                current["date"].strftime(



                    "%Y-%m-%d"



                ),







            "environmental_data": {







                "rainfall_mm":



                    round(



                        float(



                            current[



                                "rainfall_mm"



                            ]



                        ),



                        3



                    ),







                "soil_moisture":



                    round(



                        float(



                            current[



                                "soil_moisture"



                            ]



                        ),



                        4



                    ),







                "ndvi":



                    round(



                        float(



                            current["ndvi"]



                        ),



                        4



                    ),







                "temperature_c":



                    round(



                        float(



                            current[



                                "temperature_c"



                            ]



                        ),



                        2



                    ),







                "evaporation_mm":



                    round(



                        float(



                            current[



                                "evaporation_mm"



                            ]



                        ),



                        3



                    ),



            }



        })







        return prediction







    except ValueError as error:







        raise HTTPException(



            status_code=404,



            detail=str(error)



        )







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
        # --------------------------------------------------------
        # GET ALL DATA FOR DISTRICT
        # --------------------------------------------------------

        district_df = find_district(district_name)

        if district_df.empty:
            raise ValueError(
                f"District '{district_name}' was not found."
            )

        district_df = (
            district_df
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        if district_df.empty:
            raise ValueError(
                f"No historical data available for '{district_name}'."
            )

        # --------------------------------------------------------
        # MODEL NEEDS CURRENT MONTH + PREVIOUS 3 MONTHS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # SAFE FLOAT
        # --------------------------------------------------------

        def safe_float(value, decimals=4):

            if pd.isna(value):
                return None

            try:
                return round(float(value), decimals)

            except (ValueError, TypeError):
                return None

        # --------------------------------------------------------
        # BUILD HISTORICAL ML PREDICTIONS
        #
        # index 3 is the first month that has:
        # current + lag1 + lag2 + lag3
        # --------------------------------------------------------

        full_history = []

        for i in range(3, len(district_df)):

            current = district_df.iloc[i]
            lag1 = district_df.iloc[i - 1]
            lag2 = district_df.iloc[i - 2]
            lag3 = district_df.iloc[i - 3]

            # ----------------------------------------------------
            # SKIP MONTH IF REQUIRED ENVIRONMENTAL DATA IS MISSING
            # ----------------------------------------------------

            required_values = [
                current["rainfall_mm"],
                current["soil_moisture"],
                current["ndvi"],
                current["temperature_c"],
                current["evaporation_mm"],

                lag1["rainfall_mm"],
                lag2["rainfall_mm"],
                lag3["rainfall_mm"],

                lag1["soil_moisture"],
                lag2["soil_moisture"],
                lag3["soil_moisture"],

                lag1["ndvi"],
                lag2["ndvi"],
                lag3["ndvi"],

                lag1["temperature_c"],
                lag2["temperature_c"],
                lag3["temperature_c"],

                lag1["evaporation_mm"],
                lag2["evaporation_mm"],
                lag3["evaporation_mm"],
            ]

            if any(pd.isna(value) for value in required_values):
                continue

            # ----------------------------------------------------
            # SEASONAL FEATURES
            # ----------------------------------------------------

            month = int(current["date"].month)

            month_sin = np.sin(
                2 * np.pi * month / 12
            )

            month_cos = np.cos(
                2 * np.pi * month / 12
            )

            # ----------------------------------------------------
            # EXACT SAME 22 FEATURES USED BY CURRENT PREDICTION
            # ----------------------------------------------------

            features = {

                # Current environmental conditions
                "rainfall_mm":
                    float(current["rainfall_mm"]),

                "soil_moisture":
                    float(current["soil_moisture"]),

                "ndvi":
                    float(current["ndvi"]),

                "temperature_c":
                    float(current["temperature_c"]),

                "evaporation_mm":
                    float(current["evaporation_mm"]),

                # Rainfall lags
                "rainfall_mm_lag1":
                    float(lag1["rainfall_mm"]),

                "rainfall_mm_lag2":
                    float(lag2["rainfall_mm"]),

                "rainfall_mm_lag3":
                    float(lag3["rainfall_mm"]),

                # Soil moisture lags
                "soil_moisture_lag1":
                    float(lag1["soil_moisture"]),

                "soil_moisture_lag2":
                    float(lag2["soil_moisture"]),

                "soil_moisture_lag3":
                    float(lag3["soil_moisture"]),

                # NDVI lags
                "ndvi_lag1":
                    float(lag1["ndvi"]),

                "ndvi_lag2":
                    float(lag2["ndvi"]),

                "ndvi_lag3":
                    float(lag3["ndvi"]),

                # Temperature lags
                "temperature_c_lag1":
                    float(lag1["temperature_c"]),

                "temperature_c_lag2":
                    float(lag2["temperature_c"]),

                "temperature_c_lag3":
                    float(lag3["temperature_c"]),

                # Evaporation lags
                "evaporation_mm_lag1":
                    float(lag1["evaporation_mm"]),

                "evaporation_mm_lag2":
                    float(lag2["evaporation_mm"]),

                "evaporation_mm_lag3":
                    float(lag3["evaporation_mm"]),

                # Seasonal encoding
                "month_sin":
                    float(month_sin),

                "month_cos":
                    float(month_cos),
            }

            # ----------------------------------------------------
            # RUN THE REAL RANDOM FOREST MODEL
            # ----------------------------------------------------

            prediction = run_prediction(features)

            # ----------------------------------------------------
            # CREATE HISTORY RECORD
            # ----------------------------------------------------

            full_history.append({

                "date":
                    current["date"].strftime("%Y-%m-%d"),

                "year":
                    int(current["date"].year),

                "month":
                    int(current["date"].month),

                "month_label":
                    current["date"].strftime("%b"),

                "month_year":
                    current["date"].strftime("%Y-%m"),

                # Environmental values
                "rainfall_mm":
                    safe_float(
                        current["rainfall_mm"],
                        3
                    ),

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

                # REAL ML RESULTS
                "drought_probability":
                    prediction[
                        "drought_probability"
                    ],

                "drought_probability_percent":
                    prediction[
                        "drought_probability_percent"
                    ],

                "predicted_drought":
                    prediction[
                        "predicted_drought"
                    ],

                "prediction_status":
                    prediction[
                        "prediction_status"
                    ],

                "risk_level":
                    prediction[
                        "risk_level"
                    ],
            })

        # --------------------------------------------------------
        # MAKE SURE MODEL PRODUCED HISTORY
        # --------------------------------------------------------

        if not full_history:
            raise ValueError(
                f"Unable to create historical ML predictions "
                f"for '{district_name}'."
            )

        # --------------------------------------------------------
        # WEBSITE ONLY NEEDS MOST RECENT 24 MONTHS
        # --------------------------------------------------------

        history = full_history[-24:]

        latest = history[-1]

        # --------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------

        return {

            "success": True,

            "district":
                actual_district,

            "province":
                province,

            "records":
                len(history),

            "latest_date":
                latest["date"],

            "current": {

                "rainfall_mm":
                    latest["rainfall_mm"],

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



        # Get unique districts from the dataset



        district_list = (



            master_df[["province", "district"]]



            .drop_duplicates()



            .sort_values(["province", "district"])



        )







        comparison_results = []







        for _, row in district_list.iterrows():







            district_name = str(row["district"])



            province = str(row["province"])







            try:



                # IMPORTANT:



                # Reuse the already-working district prediction endpoint



                prediction = predict_district(district_name)







                comparison_results.append({



                    "district": district_name,



                    "province": province,







                    "drought_probability":



                        prediction["drought_probability"],







                    "drought_probability_percent":



                        prediction["drought_probability_percent"],







                    "predicted_drought":



                        prediction["predicted_drought"],







                    "prediction_status":



                        prediction["prediction_status"],







                    "risk_level":



                        prediction["risk_level"],







                    "data_date":



                        prediction.get("data_date"),







                    "environmental_data":



                        prediction.get(



                            "environmental_data",



                            {}



                        )



                })







            except Exception as district_error:







                print(



                    f"ERROR predicting {district_name}: "



                    f"{district_error}"



                )







                continue







        # Highest risk first



        comparison_results.sort(



            key=lambda x:



                x["drought_probability"],



            reverse=True



        )







        return {



            "success": True,



            "count": len(comparison_results),



            "warning_threshold": WARNING_THRESHOLD,



            "districts": comparison_results



        }







    except Exception as error:







        print("COMPARE DISTRICTS ERROR:", error)







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







    try:



        base_dir = os.path.dirname(os.path.abspath(__file__))







        # ----------------------------------------------------



        # FILE PATHS



        # ----------------------------------------------------







        risk_path = os.path.join(



            base_dir,



            "data",



            "latest_district_risk.csv"



        )







        coordinates_path = os.path.join(



            base_dir,



            "data",



            "district_coordinates.csv"



        )







        # ----------------------------------------------------



        # LOAD DATA



        # ----------------------------------------------------







        risk_df = pd.read_csv(risk_path)



        coords_df = pd.read_csv(coordinates_path)







        # Clean names before merging



        risk_df["district"] = (



            risk_df["district"]



            .astype(str)



            .str.strip()



        )







        risk_df["province"] = (



            risk_df["province"]



            .astype(str)



            .str.strip()



        )







        coords_df["district"] = (



            coords_df["district"]



            .astype(str)



            .str.strip()



        )







        coords_df["province"] = (



            coords_df["province"]



            .astype(str)



            .str.strip()



        )







        # ----------------------------------------------------



        # MERGE ML RESULTS WITH COORDINATES



        # ----------------------------------------------------







        map_df = risk_df.merge(



            coords_df[



                [



                    "province",



                    "district",



                    "latitude",



                    "longitude"



                ]



            ],



            on=["province", "district"],



            how="left"



        )







        # ----------------------------------------------------



        # REMOVE RECORDS WITHOUT COORDINATES



        # ----------------------------------------------------







        map_df = map_df.dropna(



            subset=["latitude", "longitude"]



        )







        # ----------------------------------------------------



        # CONVERT DATA FOR JSON



        # ----------------------------------------------------







        districts = []







        for _, row in map_df.iterrows():







            probability = float(row["drought_probability"])







            # Use risk level already created by ML pipeline



            risk_level = str(row["risk_level"])







            district_data = {







                "district": str(row["district"]),



                "province": str(row["province"]),







                "latitude": round(



                    float(row["latitude"]),



                    6



                ),







                "longitude": round(



                    float(row["longitude"]),



                    6



                ),







                "drought_probability": round(



                    probability,



                    4



                ),







                "drought_probability_percent": round(



                    probability * 100,



                    2



                ),







                "risk_level": risk_level,







                "environmental_data": {







                    "rainfall_mm": round(



                        float(row["rainfall_mm"]),



                        3



                    ),







                    "soil_moisture": round(



                        float(row["soil_moisture"]),



                        4



                    ),







                    "ndvi": round(



                        float(row["ndvi"]),



                        4



                    ),







                    "temperature_c": round(



                        float(row["temperature_c"]),



                        2



                    ),







                    "evaporation_mm": round(



                        float(row["evaporation_mm"]),



                        3



                    )



                }



            }







            districts.append(district_data)







        # Highest risk first



        districts = sorted(



            districts,



            key=lambda x: x["drought_probability"],



            reverse=True



        )







        # ----------------------------------------------------



        # RETURN RESPONSE



        # ----------------------------------------------------







        return {







            "success": True,







            "count": len(districts),







            "total_risk_records": len(risk_df),







            "coordinates_found": len(districts),







            "districts": districts



        }







    except Exception as e:







        return {



            "success": False,



            "error": str(e)



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
        coords_df["district"] = coords_df["district"].astype(str).str.strip()
        coords_df["province"] = coords_df["province"].astype(str).str.strip()

        match = coords_df[
            (coords_df["district"].str.lower() == actual_district.lower()) &
            (coords_df["province"].str.lower() == province.lower())
        ]

        # Province labels can differ slightly between datasets, so fall back
        # to a unique district-name match before failing.
        if match.empty:
            match = coords_df[
                coords_df["district"].str.lower() == actual_district.lower()
            ]

        if match.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Coordinates not found for '{actual_district}'."
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
            "drought_probability": prediction.get("drought_probability"),
            "drought_probability_percent": prediction.get("drought_probability_percent"),
            "predicted_drought": prediction.get("predicted_drought"),
            "prediction_status": prediction.get("prediction_status"),
            "risk_level": prediction.get("risk_level"),
            "warning_threshold": prediction.get("warning_threshold"),
            "environmental_data": {
                "rainfall_mm": env.get("rainfall_mm"),
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