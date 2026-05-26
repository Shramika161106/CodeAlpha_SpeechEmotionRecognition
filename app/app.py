import streamlit as st
import librosa
import numpy as np
import joblib
import tensorflow as tf
import tempfile


# Page settings
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️"
)

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV audio file and predict emotion.")


# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["ANN", "CNN", "LSTM"]
)


# Load model + encoder + scaler
if model_choice == "ANN":
    model = tf.keras.models.load_model(
        "models/emotion_model.h5"
    )
    encoder = joblib.load(
        "models/encoder.pkl"
    )
    scaler = joblib.load(
        "models/scaler.pkl"
    )

elif model_choice == "CNN":
    model = tf.keras.models.load_model(
        "models/cnn_emotion_model.h5"
    )
    encoder = joblib.load(
        "models/encoder_cnn.pkl"
    )
    scaler = joblib.load(
        "models/scaler_cnn.pkl"
    )

else:
    model = tf.keras.models.load_model(
        "models/lstm_emotion_model.h5"
    )
    encoder = joblib.load(
        "models/encoder_lstm.pkl"
    )
    scaler = joblib.load(
        "models/scaler_lstm.pkl"
    )


# Feature extraction
def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc_scaled


uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(
            uploaded_file.read()
        )

        features = extract_features(
            temp_audio.name
        )

    features = scaler.transform(
        [features]
    )

    if model_choice in ["CNN", "LSTM"]:
        features = features.reshape(
            features.shape[0],
            features.shape[1],
            1
        )

    prediction = model.predict(
        features
    )

    predicted_index = np.argmax(
        prediction
    )

    emotion = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = np.max(
        prediction
    ) * 100

    st.success(
        f"Predicted Emotion: {emotion}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )