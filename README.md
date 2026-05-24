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

## Dataset Used

RAVDESS Dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)

The dataset contains emotional speech recordings from multiple actors.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- MFCC Feature Extraction

---

## Model Architecture

The project uses:

- MFCC Feature Extraction
- Dense Neural Network (DNN)
- Dropout Layers
- Softmax Classification

---

## Accuracy

Model Accuracy Achieved:

**74.55%**

---

## Project Structure

```text
Speech-Emotion-Recognition/
│
├── dataset/
├── src/
│   ├── feature_extraction.py
│   ├── dataset_loader.py
│   ├── train.py
│   └── predict.py
│
├── models/
├── requirements.txt
├── README.md
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
