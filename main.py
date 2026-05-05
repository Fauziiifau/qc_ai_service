from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="QC Defect AI Service")
model_pipeline = joblib.load('model_defect_v2.pkl')

class DefectData(BaseModel):
    proses: str
    jenis_defect: str

@app.post("/predict-action")
def predict_action(data: DefectData):
    try:
        print(f"Defect yang diterima = '{data.jenis_defect}'")
        print(f"Proses yang diterima = '{data.proses}'")

        input_data = pd.DataFrame({
            'Jenis defect': [data.jenis_defect],
            'Proses': [data.proses]
        })

        prediksi = model_pipeline.predict(input_data)
        
        hasil_teks = str(prediksi[0]).replace("[", "").replace("]", "").replace("'", "")
        return {
            "status": "success",
            "rekomendasi": hasil_teks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))