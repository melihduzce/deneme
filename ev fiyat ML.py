import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Veriyi yükle
data = pd.read_csv("melb_data.csv")

# 2. Hedef değişken ve kullanılacak özellikler
data = data.dropna()  # eksik verileri temizle
y = data["Price"]

# Basit ve sayısal bazı özellikleri kullanalım
features = ["Rooms", "Distance", "Landsize", "BuildingArea", "YearBuilt"]
X = data[features]

# 3. Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 4. Model oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Tahmin yap
preds = model.predict(X_test)

# 6. Başarıyı ölç
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.2f}")
