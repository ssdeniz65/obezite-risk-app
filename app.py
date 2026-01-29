import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# Modeli yükle
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

# Özellik listesi
features = [
    "BASE_LINE_WEIGHT", "BMI", "AGE", "BECK_DEPRESSION_SCALE",
    "EATING_STYLE_HUNGER_SCORE", "FOOD_CHOISE_SCORE",
    "PHYSICAL_ACTIVITY_SCORE", "STATUS", "GENDER"
]

st.set_page_config(layout="centered")
st.title("🎯 Geri Kilo Alma Risk Tahmini")
st.markdown("Klinik verilerle kişiye özel risk analizi ve SHAP katkı görselleştirmesi.")

# Form arayüzü
with st.form("risk_form"):
    weight = st.slider("Başlangıç Kilosu (kg)", 50, 150, 92)
    bmi = st.slider("BMI", 18, 45, 31)
    age = st.slider("Yaş", 18, 80, 40)
    depression = st.slider("Beck Depresyon Skoru", 0, 63, 26)
    hunger = st.slider("Açlık Tipi Yeme Skoru", 0, 100, 80)
    food = st.slider("Besin Tercihi Skoru", 0, 100, 88)
    activity = st.slider("Fiziksel Aktivite Skoru", 0, 100, 40)
    status = st.selectbox("Medeni Durum", ["Bekar", "Evli"])
    gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    submitted = st.form_submit_button("Tahmin Et")

# Tahmin işlemleri
if submitted:
    input_data = pd.DataFrame([{
        "BASE_LINE_WEIGHT": weight,
        "BMI": bmi,
        "AGE": age,
        "BECK_DEPRESSION_SCALE": depression,
        "EATING_STYLE_HUNGER_SCORE": hunger,
        "FOOD_CHOISE_SCORE": food,
        "PHYSICAL_ACTIVITY_SCORE": activity,
        "STATUS": 1 if status == "Evli" else 0,
        "GENDER": 1 if gender == "Erkek" else 0
    }])

    prob = model.predict_proba(input_data)[0][1]
    st.success(f"📊 Geri kilo alma olasılığı: **%{prob*100:.2f}**")

    # Klinik öneri fonksiyonu
    def get_clinical_advice(risk):
        if risk >= 0.85:
            return (
                "🔴 ML Önerisi: Yüksek risk – Yakın takip önerilir.\n"
                "📅 Takip planı: Haftalık kontroller, psikolojik destek önerilir."
            )
        elif risk >= 0.65:
            return (
                "🟠 ML Önerisi: Orta risk – Davranışsal müdahale önerilir.\n"
                "📅 Takip planı: 2 haftada bir izlem ve diyet eğitimi."
            )
        else:
            return (
                "🟢 ML Önerisi: Düşük risk.\n"
                "📅 Takip planı: Rutin ayda bir kontrol yeterli."
            )

    st.subheader("🩺 Klinik Karar Yardımı")
    st.info(get_clinical_advice(prob))

    # SHAP Analizi
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("🧠 SHAP Özellik Katkı Analizi")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
