import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Sadə data (Sahə -> Qiymət)
data = {
    "sahe": [50, 60, 70, 85, 100, 120],
    "qiymet": [70, 85, 95, 120, 140, 165],
}
df = pd.DataFrame(data)
X = df[["sahe"]]
y = df["qiymet"]

# 2. Modeli öyrətmək
model = LinearRegression()
model.fit(X, y)
print("Model uğurla öyrədildi!")

# 3. Test edək
test_sahe = [[80]]
tahmin = model.predict(test_sahe)
print(f"80 kv.m ev üçün təxmini qiymət: {tahmin[0]:.2f} min AZN")

# 4. Modeli fayl kimi yadda saxlayırıq
joblib.dump(model, "model.pkl")
print("Model 'model.pkl' olaraq yadda saxlanıldı.")