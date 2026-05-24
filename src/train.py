import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from feature_extraction import extract_mfcc
from dataset_loader import get_emotion


X = []
y = []

dataset_path = "../dataset"

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion = get_emotion(file)

            features = extract_mfcc(file_path)

            X.append(features)
            y.append(emotion)


X = np.array(X)
y = np.array(y)


# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")


# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save encoder
joblib.dump(encoder, "../models/encoder.pkl")

y_categorical = to_categorical(y_encoded)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Model
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(40,)))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation='softmax'))


# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32
)


# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy * 100)


# Save model
model.save("../models/emotion_model.h5")