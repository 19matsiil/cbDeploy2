import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_excel("datasetknnfix(09020624047).xlsx")
    return df

@st.cache_data
def preprocess_and_train(df):
    df = df.copy()

    # One-hot encoding untuk data kategorikal
    df = pd.get_dummies(df, columns=["gender", "smoking_history"])

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy, X.columns

def main():
    st.set_page_config(page_title="Prediksi Diabetes", layout="centered")
    st.title("🩺 Prediksi Diabetes")

    df = load_data()
    model, accuracy, feature_columns = preprocess_and_train(df)

    st.subheader("Input Data Pasien")

    age = st.slider("Age", 1, 100, 30)
    hypertension = st.slider("Hipertensi (0 = Tidak, 1 = Ya)", 0, 1, 0)
    heart_disease = st.slider("Penyakit Hati (0 = Tidak, 1 = Ya)", 0, 1, 0)
    bmi = st.slider("BMI(Indeks Massa Tubuh)", 10.0, 50.0, 25.0, step=0.1)
    hba1c = st.slider("HbA1c Level(rata-rata Hemoglobin)", 3.0, 15.0, 5.5, step=0.1)
    glucose = st.slider("Kadar Gula Darah", 50, 300, 120)

    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking History", df["smoking_history"].unique())

    if st.button("Prediksi"):
        input_dict = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }

        for col in feature_columns:
            if col not in input_dict:
                input_dict[col] = 0

        gender_col = f"gender_{gender}"
        smoking_col = f"smoking_history_{smoking}"

        if gender_col in input_dict:
            input_dict[gender_col] = 1

        if smoking_col in input_dict:
            input_dict[smoking_col] = 1

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ Berpotensi Diabetes")
        else:
            st.success("✅ Tidak Diabetes")

        st.info(f"Akurasi Model: {accuracy*100:.2f}%")

    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(df)

if __name__ == "__main__":
    main()