#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:59:44 2024

@author: eren
"""

"""
Bu proje, önemli bir sağlık ve zindelik sorununu ele almak için veri bilimi 
tekniklerinden yararlanarak kalori yakma tahmini için öngörücü bir model geliştirmeyi 
amaçlamaktadır. Fiziksel aktivite türü, süresi, yoğunluğu ve yaş, kilo ve cinsiyet gibi 
bireysel özellikler gibi çeşitli girdi özelliklerini analiz ederek, belirli bir aktivite 
sırasında yakılan kalori miktarını doğru bir şekilde tahmin etmek amaçlanmaktadır. 
Bu model, kalori yönetimi ve fiziksel aktivite planlamasını optimize etmek için 
bireyler, fitness meraklıları ve sağlık uzmanları için değerli bir araç olarak hizmet 
edecektir.

"""



# Kütüphaneleri ve Verileri Import etme.
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib




df = pd.read_csv("/mnt/c/Users/CASPER/OneDrive/Masaüstü/Caloriees Burnt Prediction/exercise.csv").copy()
y = pd.read_csv("/mnt/c/Users/CASPER/OneDrive/Masaüstü/Caloriees Burnt Prediction/calories.csv").copy()


df['Calories'] = y['Calories']

# Veritabanını tanıyalım.
print(df.head(10))

print(df.info())

print(df.describe())

print(df.select_dtypes(include=['object']).describe())

print(df.shape)



# df Analizi.
numerical_data = df.select_dtypes(include='number').drop('User_ID',axis=1)
numerical_data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# Boxplot çizdirme
plt.figure(figsize=(10, 8))
sns.boxplot(numerical_data)
plt.xticks(rotation=45)
plt.show()

# Veri Setindeki Cinsiyet Sayısını çizdirme
categorical_data = df.select_dtypes(exclude='number')
for column in categorical_data.columns:
    sns.countplot(data=categorical_data, x=column, palette="Set1")
    plt.title(f"Countplot of {column}")
    plt.show()


# Sütunların analizi
sns.pairplot(numerical_data,corner = True)
plt.show()


# Heatmap ile Korelasyon Analizi
corr = numerical_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,annot=True, cmap='coolwarm')
plt.show()



# ***** Data Cleaning *****

# Gereksiz Sütunları Kaldırma
df.drop(['User_ID'], axis=1, inplace=True)


# Yinelenen Satırların Kaldırılması
duplicate_rows = df.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")
df.drop_duplicates(inplace=True)


# Eksik Verilerin İşlenmesi
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(10))

total = df.isnull().sum().sum()
print('Total Null values =' ,total)


# ***** Data Preprocessing *****

# Kategorik Değişkenlerin Kodlanması
print(df['Gender'].value_counts())
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

"""
# Sayısal sütunlara log dönüşümü
df = np.log1p(df)
"""

# Veri Bölme
X = df.drop(['Calories'], axis=1)
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=44)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size=.5,random_state=44)


# Veri Normalizasyonu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# scaler'ı kaydet
joblib.dump(scaler, 'scaler.pkl')


# Farklı Modeller Kullanarak Hangisinde Daha İyi Sonuç Alıcaz Görmek İstiyorum.
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(random_state=42)),
    ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42)),
    ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
    ('K-Nearest Neighbors Regressor', KNeighborsRegressor()),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42))
]

model_performance = []
for reg_name, reg in regressors:
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv_scores = cross_val_score(reg, X_train, y_train, cv=5)
    model_performance.append((reg_name, r2, mse, mae, cv_scores))
    print(f'{reg_name}:')
    print("Cross Val Score: ", cv_scores.mean())
    print('R2 Score: ', r2)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print('------------------------------------')
    


# Model Performansını Görselleştirme
performance_df = pd.DataFrame(model_performance, columns=['Model', 'R2', 'MSE', 'MAE', 'Cross-Val Score'])
performance_df.set_index('Model', inplace=True)


fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# R2 Skoru
performance_df['R2'].plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8, title="R2 Skorları")
axes[0].set_ylabel("R2 Skoru")

# MSE
performance_df['MSE'].plot(kind='bar', ax=axes[1], color='salmon', alpha=0.8, title="MSE Değerleri")
axes[1].set_ylabel("MSE")

# MAE
performance_df['MAE'].plot(kind='bar', ax=axes[2], color='limegreen', alpha=0.8, title="MAE Değerleri")
axes[2].set_ylabel("MAE")
axes[2].set_xlabel("Modeller")

plt.tight_layout()
plt.show()



# ***** Hiperparametre Ayarı *****
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# En İyi Modelin Özellik Önemini Görselleştirme
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', title="Özellik Önemi")
plt.show()

# ***** Modeli Kaydetme ve Kullanma *****
joblib.dump(best_rf_model, 'calorie_prediction_model.pkl')
loaded_model = joblib.load('calorie_prediction_model.pkl')




# Şimdi rastgele bir veri ile deneyelim.

# Rastgele bir veri örneği oluştur
random_data = pd.DataFrame({
    'Gender': [1],  # Label Encoding'den sonra 'male' -> 1
    'Age': [30],
    'Height': [175.0],
    'Weight': [70.0],
    'Duration': [45.0],
    'Heart_Rate': [120.0],
    'Body_Temp': [37.0]
})

# Rastgele veriyi aynı scaler ile ölçeklendir
random_data_scaled = scaler.transform(random_data)

# En iyi performansı gösteren modelle tahmin yap
best_model = RandomForestRegressor(random_state=42)  # Buraya en iyi modeli koy
best_model.fit(X_train, y_train)  # Eğitim setiyle modeli eğit
calories_prediction = best_model.predict(random_data_scaled)

print(f"Tahmin Edilen Kalori: {calories_prediction[0]:.2f} kcal")



