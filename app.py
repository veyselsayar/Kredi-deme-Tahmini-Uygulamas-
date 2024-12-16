import streamlit as st
import numpy as np
from joblib import load

# Modeli yükleme
model = load("eniyi.joblib")

# Sayfa başlığı
st.title("Kredi Ödeme Tahmini Uygulaması")

st.write("Müşterinin kredi ödemesi yapıp yapmayacağını tahmin edin!")

# Kullanıcıdan giriş al
gelir = st.number_input("Gelir (TL)", min_value=0, step=1000)
yas = st.number_input("Yaş", min_value=18, max_value=100, step=1)
kredibilite_skoru = st.number_input("Kredibilite Skoru", min_value=0, max_value=1000, step=10)
egitim_seviyesi = st.selectbox("Eğitim Seviyesi", ["Lisans", "Lise", "Ortaokul"])
ev_durumu = st.selectbox("Ev Durumu", ["Kiracı", "Ev Sahibi"])
odeme_gecmisi = st.selectbox("Ödeme Geçmişi", ["Düzgün", "Problemli"])

# Girişleri işleme
def encode_input(egitim, ev, odeme):
    # LabelEncoder ile yapılan işlemleri manuel olarak gerçekleştiriyoruz
    egitim_mapping = {"Lisans": 2, "Lise": 1, "Ortaokul": 0}
    ev_mapping = {"Kiracı": 0, "Ev Sahibi": 1}
    odeme_mapping = {"Düzgün": 0, "Problemli": 1}
    return egitim_mapping[egitim], ev_mapping[ev], odeme_mapping[odeme]

encoded_egitim, encoded_ev, encoded_odeme = encode_input(egitim_seviyesi, ev_durumu, odeme_gecmisi)

# Tahmin butonu
if st.button("Tahmin Yap"):
    # Model için özellik vektörü oluştur
    features = np.array([[gelir, yas, kredibilite_skoru, encoded_egitim, encoded_ev, encoded_odeme]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]  # Ödeme yapma olasılığı

    # Sonuç gösterimi
    if prediction[0] == 1:
        st.success(f"Kredi ödemesi yapma ihtimali yüksek! (%{probability*100:.2f})")
    else:
        st.error(f"Kredi ödemesi yapmama ihtimali yüksek! (%{(1-probability)*100:.2f})")
