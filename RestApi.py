from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import sqlite3
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel

from make_prediction import Predictor

#python -m uvicorn RestApi:app --port 8001

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "real_estate.db")

app = FastAPI()

@app.get("/predict", response_class=HTMLResponse)
def predict_page():
    return (BASE_DIR / "predict.html").read_text(encoding="utf-8")

class PredictRequest(BaseModel):
    city: str
    rooms: int
    squareMeters: float
    centreDistance: float
    hasParkingSpace: bool
    hasElevator: bool
    hasSecurity: bool

predictor = Predictor()

@app.post("/api/predict")
def predict(req: PredictRequest):

    payload = req.dict()

    payload["hasParkingSpace"] = 1 if req.hasParkingSpace else 0
    payload["hasElevator"]     = 1 if req.hasElevator else 0
    payload["hasSecurity"]     = 1 if req.hasSecurity else 0

    payload["city"] = payload["city"].strip().lower()

    pred = predictor.make_prediction(payload)

    return {
        "prediction_pln": pred,
        "metrics": predictor.show_metrics()
    }

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")

def query_labels_values(sql: str, params: Dict[str, Any] | None = None) -> Dict[str, List[Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or {})
        rows = cur.fetchall()
    finally:
        conn.close()

    labels = [str(r[0]) for r in rows]
    values = [float(r[1]) for r in rows]
    return {"labels": labels, "values": values}


@app.get("/api/charts/avg-price-by-rooms")
def avg_price_by_rooms():
    sql = """
    SELECT rooms AS label, AVG(price) AS value
    FROM poland_rent
    GROUP BY rooms
    ORDER BY rooms;
    """
    return query_labels_values(sql)

@app.get("/api/charts/approximately-sqms")
def approximately_sqms():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT squareMeters FROM poland_rent;", conn)
    finally:
        conn.close()

    # python_json = {
    #     "0-49": (df.squareMeters < 50),
    #     "50-99": (df.squareMeters >= 50) & (df.squareMeters < 100),
    #     "100-149": (df.squareMeters >= 100) & (df.squareMeters < 150),
    #     "150+": (df.squareMeters >= 150),
    # }

    python_json = {
        "0-34": (df.squareMeters < 35),
        "35-69": (df.squareMeters >= 35) & (df.squareMeters < 70),
        "70-104": (df.squareMeters >= 70) & (df.squareMeters < 105),
        "105-144": (df.squareMeters >= 105) & (df.squareMeters < 145),
        "145+": (df.squareMeters >= 145),
    }

    labels = []
    values = []

    for label, condition in python_json.items():
        labels.append(label)
        values.append(int(condition.sum()))

    return {
        "labels": labels,
        "values": values
    }

@app.get("/api/charts/avg-price-by-region")
def avg_price_by_region():

    sql = """
      SELECT city AS label, AVG(price) AS value
      FROM poland_rent
      GROUP BY city
      ORDER BY value DESC;
      """
    return query_labels_values(sql)

@app.get("/api/charts/parkinglot-percentage-per-region")
def parking_share():
    sql = """
    SELECT
      CASE WHEN hasParkingSpace = 1 THEN 'With parking' ELSE 'Without parking' END AS label,
      COUNT(*) AS value
    FROM poland_rent
    GROUP BY label
    ORDER BY label;
    """
    return query_labels_values(sql)


@app.get("/api/charts/avg-price-by-month")
def avg_price_by_month():
    sql = """
    SELECT
      substr(date, 1, 7) AS label,
      AVG(price) AS value
    FROM poland_rent
    WHERE date >= '2023-11-01' AND date <= '2024-06-01'
    GROUP BY label
    ORDER BY label;
    """
    return query_labels_values(sql)

