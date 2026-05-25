# Speech Emotion Recognition System

## Project Overview

This project recognizes human emotions from speech audio using Machine Learning and Deep Learning techniques.

The system extracts audio features using MFCC (Mel-Frequency Cepstral Coefficients) and predicts emotions such as:

- Happy
- Sad
- Angry
- Calm
- Fearful
- Neutral
- Disgust
- Surprised

---

## Objective

To recognize human emotions from speech audio using speech signal processing and deep learning models.

---

## Dataset

This project uses the RAVDESS dataset for speech emotion recognition.

Download dataset from:
https://zenodo.org/record/1188976

After downloading, place it inside:

dataset/

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- MFCC Feature Extraction

---

## Model Architecture

## Model Architectures Implemented

This project compares multiple deep learning approaches for speech emotion recognition:

### 1. ANN (Artificial Neural Network / Dense Neural Network)

- MFCC feature extraction
- Dense layers with dropout
- Best performing model

### 2. CNN (Convolutional Neural Network)

- 1D convolution over MFCC feature representation
- Moderate performance

### 3. LSTM (Long Short-Term Memory Network)

- Sequence-based model tested on MFCC features
- Used for temporal speech pattern modeling

---

## Model Performance Comparison

| Model | Accuracy |
|--------|----------|
| ANN (Dense NN) | ~74.55% |
| CNN | ~65.95% |
| LSTM | ~35% |

### Observation

ANN achieved the highest performance on the extracted MFCC feature representation.

CNN showed moderate performance.

LSTM underperformed because MFCC features were averaged into fixed-length vectors, reducing temporal sequence information required for effective recurrent modeling.

---

## Future Improvements

- Preserve MFCC time sequences for better LSTM performance
- Multi-dataset training (RAVDESS + TESS + EMO-DB)
- Real-time microphone emotion detection
- Streamlit web deployment
- Hyperparameter tuning

## Accuracy

Model Accuracy Achieved:

**74.55%**

---

## Project Structure

```text
Speech-Emotion-Recognition/
│
├── app/
├── dataset/
├── models/
│   ├── emotion_model.h5
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── src/
│   ├── feature_extraction.py
│   ├── dataset_loader.py
│   ├── train.py
│   ├── train_cnn.py
│   ├── train_lstm.py
│   └── predict.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```
## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
cd src
python train.py
```

### 3. Predict emotion

```bash
python predict.py
```

---

## Future Improvements

- Real-time microphone emotion detection
- Streamlit Web Application
- CNN/LSTM implementation
- Better accuracy optimization
- Multilingual speech emotion recognition 





