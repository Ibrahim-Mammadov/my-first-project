import joblib
from fastapi import FastAPI
import pandas as pd

# 1. FastAPI tətbiqini başladırıq
app = FastAPI()

# 2. İndicə yaratdığımız model.pkl faylını bura yükləyirik
model = joblib.load("model.pkl")


# 3. Ana səhifə üçün sadə bir salamlaşma
@app.get("/")
def ana_sehire():
    return {"mesaj": "Ev Qiyməti Təxmin API-ı aktivdir!"}


# 4. Təxmin edən əsas qapı (Endpoint)
# İstifadəçi brauzerdə məsələn /texmin et?sahe=80 yazacaq
@app.get("/texminet")
def tahmin_et(sahe: float):
    # Giriş datamızı modelin başa düşəcəyi formata (DataFrame) salırıq
    yeni_data = pd.DataFrame([[sahe]], columns=["sahe"])

    # Modeldən təxmini alırıq
    tahmin = model.predict(yeni_data)

    # Nəticəni istifadəçiyə JSON (formatında) geri qaytarırıq
    return {"ev_sahasi_kv_m": sahe, "tahmini_qiymet_min_azn": round(tahmin[0], 2)}