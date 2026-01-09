import pandas as pd
import numpy as np
import joblib
from pathlib import Path
#from model import city_categories
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent

class Predictor:
    def __init__(self):
        self.model = joblib.load(BASE_DIR / "model.joblib")
        self.encoder = joblib.load(BASE_DIR / "encoder.joblib")
        self.scalar_x = joblib.load(BASE_DIR / "scaler_x.joblib")
        self.scalar_y = joblib.load(BASE_DIR / "scaler_y.joblib")
        self.featured_columns = joblib.load(BASE_DIR / "featured_columns.joblib")
        self.metrics = joblib.load(BASE_DIR / "metrics.joblib")
        self.pred_pln = 0

    def make_prediction(self, json):
        json = dict(json)

        required = [
            "city", "rooms", "squareMeters", "centreDistance",
            "hasParkingSpace", "hasElevator", "hasSecurity"
        ]
        missing = [k for k in required if k not in json]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        city_value = str(json["city"]).strip().lower()
        sqmt_value = float(json["squareMeters"])
        centre_dis = float(json["centreDistance"])

        scaled_df = pd.DataFrame([{
            "squareMeters": sqmt_value,
            "centreDistance": centre_dis
        }])
        scaled = self.scalar_x.transform(scaled_df)
        sqmt_scaled = float(scaled[0, 0])
        centre_scaled = float(scaled[0, 1])

        centre_scaled *= 0.25

        city_df = pd.DataFrame({"city": [city_value]})
        city_ohe = self.encoder.transform(city_df)

        city_ohe_df = pd.DataFrame(
            city_ohe,
            columns=self.encoder.get_feature_names_out(["city"])
        )

        base = pd.DataFrame([{
            "rooms": int(json["rooms"]),
            "squareMeters": sqmt_scaled,
            "centreDistance": centre_scaled,
            "hasParkingSpace": int(json["hasParkingSpace"]),
            "hasElevator": int(json["hasElevator"]),
            "hasSecurity": int(json["hasSecurity"]),
        }])

        base = pd.concat([base.reset_index(drop=True), city_ohe_df.reset_index(drop=True)], axis=1)

        base = base.reindex(columns=self.featured_columns, fill_value=0.0)

        pred_scaled = self.model.predict(base)

        self.pred_pln = self.scalar_y.inverse_transform(
            np.array(pred_scaled).reshape(-1, 1)
        )[0, 0]

        return float(self.pred_pln)

    def show_metrics(self):
        return self.metrics
