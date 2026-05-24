import numpy as np
import joblib

from tensorflow.keras.models import load_model
from feature_extraction import extract_mfcc


# Load model
model = load_model("../models/emotion_model.h5")

# Load scaler + encoder
scaler = joblib.load("../models/scaler.pkl")
encoder = joblib.load("../models/encoder.pkl")


# Choose audio file
audio_file = "../dataset/Actor_22/03-01-02-02-02-01-22.wav"


# Extract features
features = extract_mfcc(audio_file)

features = np.array(features).reshape(1, -1)

# Scale exactly like training
features = scaler.transform(features)


# Predict probabilities
prediction = model.predict(features)

predicted_index = np.argmax(prediction)

# Decode label correctly
predicted_emotion = encoder.inverse_transform([predicted_index])


print("Predicted Emotion:", predicted_emotion[0])

print("Confidence Scores:")
for emotion, score in zip(
    encoder.classes_,
    prediction[0]
):
    print(f"{emotion}: {score:.4f}")