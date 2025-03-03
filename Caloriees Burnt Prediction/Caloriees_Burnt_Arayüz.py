#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 01:25:28 2024

@author: eren
"""

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib


# Arayüz için scaler ve model yükleme
scaler = joblib.load('scaler.pkl')
loaded_model = joblib.load('calorie_prediction_model.pkl')


# Tahmin fonksiyonu
def predict_calories():
    try:
        # Kullanıcıdan verileri al
        gender = 1 if gender_var.get() == "Male" else 0
        age = int(age_entry.get())
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        duration = float(duration_entry.get())
        heart_rate = float(heart_rate_entry.get())
        body_temp = float(body_temp_entry.get())
        
        # Girdileri bir DataFrame'e dönüştür
        user_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp]
        })
        
        # Veriyi ölçeklendir
        scaled_data = scaler.transform(user_data)
        
        # Tahmin yap
        prediction = loaded_model.predict(scaled_data)
        messagebox.showinfo("Tahmin Sonucu", f"Tahmin Edilen Kalori: {prediction[0]:.2f} kcal")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")



# Tkinter arayüzü oluştur
window = tk.Tk()
window.title("Kalori Tahmini")
window.geometry("400x400")



# Etiket ve giriş kutuları
tk.Label(window, text="Cinsiyet:").grid(row=0, column=0, pady=5)
gender_var = tk.StringVar(value="Male")
tk.OptionMenu(window, gender_var, "Male", "Female").grid(row=0, column=1)

tk.Label(window, text="Yas:").grid(row=1, column=0, pady=5)
age_entry = tk.Entry(window)
age_entry.grid(row=1, column=1)

tk.Label(window, text="Boy (cm):").grid(row=2, column=0, pady=5)
height_entry = tk.Entry(window)
height_entry.grid(row=2, column=1)

tk.Label(window, text="Kilo (kg):").grid(row=3, column=0, pady=5)
weight_entry = tk.Entry(window)
weight_entry.grid(row=3, column=1)

tk.Label(window, text="Süre (dk):").grid(row=4, column=0, pady=5)
duration_entry = tk.Entry(window)
duration_entry.grid(row=4, column=1)

tk.Label(window, text="Kalp Hizi (bpm):").grid(row=5, column=0, pady=5)
heart_rate_entry = tk.Entry(window)
heart_rate_entry.grid(row=5, column=1)

tk.Label(window, text="Vucut Sicakligi (°C):").grid(row=6, column=0, pady=5)
body_temp_entry = tk.Entry(window)
body_temp_entry.grid(row=6, column=1)



# Tahmin butonu
predict_button = tk.Button(window, text="Kalori Tahmini Yap", command=predict_calories)
predict_button.grid(row=7, columnspan=2, pady=20)


# Tkinter döngüsü
window.mainloop()

