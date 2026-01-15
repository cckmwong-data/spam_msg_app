# Accessible Spam Message Classifier with Auditory Feedback

A machine learning–based SMS spam detection application that classifies messages as spam or ham (not spam) using Natural Language Processing (NLP) and a Support Vector Machine (SVM) model, deployed via Streamlit with optional text-to-speech output for accessibility.

<img src="./images/app.png" width="" height="500">

## Highlights
- End-to-end workflow from preprocessing to deployment.
- TF-IDF vectorization for text feature extraction.
- SVM classifier optimized and stratifying train-test split for imbalanced data.
- Exported model and vectorizer artifacts for reproducible inference.
- Streamlit application with accessible user interaction and audio output.

## Skills Demonstrated
✔ NLP and Feature Engineering

✔ Supervised Machine Learning (SVM)

✔ Model Evaluation and Metric Selection

✔ Imbalanced Classification Strategies

✔ Model Serialization and Deployment

✔ Streamlit Application Development

✔ Accessibility-Aware UX Design

## Problem Statement
Users continue to receive fraudulent and malicious SMS messages. Manually screening them is inefficient and exposes individuals to privacy and financial risks. This project addresses the need for an automated and accessible solution capable of classifying SMS messages reliably.

## Project Overview

> **Note:**  
> *This Streamlit application is hosted on the free Tier of Streamlit Community Cloud. If the app has been idle for more than 24 hours, it may take some time to reactivate. In such cases, please click “xxxxx” to relaunch the application. Thank you for your patience.*

This project implements a complete machine learning workflow including:
1. Dataset acquisition (SMS spam dataset from Kaggle)
2. Text normalization and preprocessing
3. Exploratory data analysis
4. TF-IDF vectorization for numeric representation
5. SVM model training with hyperparameter tuning
6. F1-score optimization due to class imbalance
7. Export of trained classifier and vectorizer
8. Streamlit deployment with optional auditory feedback

## Key Insights & Impacts
- Reduces user exposure to fraudulent SMS content by providing fast and automated classification.
- Eliminates the need for manual message screening, improving scalability for consumer applications and communication platforms.
- Optional text-to-speech feature enhances accessibility for visually impaired users or hands-free scenarios, demonstrating inclusive design considerations.
- Provides instant and interpretable feedback, increasing user awareness of spam indicators and digital communication safety.
- Demonstrates how lightweight NLP models can deliver meaningful value without extensive infrastructure.

## Results Summary
- **Classifier:** Support Vector Machine (SVM)
- **Feature Extraction:** TF-IDF Vectorization
- **Primary Metric:** F1-Score (selected due to class imbalance)
- **Outcome:** SVM achieved highest validation F1-score among evaluated configurations
- **Artifacts Exported:** `svm_spam_model.pkl`, `tfidf_vectorizer.pkl`

## Stages of Development
1. Data Loading and Inspection  
2. Text Preprocessing and Normalization  
3. Exploratory Data Analysis  
4. TF-IDF Feature Engineering  
5. Model Training and Hyperparameter Tuning  
6. Model Validation and Metric Comparison  
7. Artifact Serialization  
8. Streamlit Application Development  
9. Accessibility Enhancement (Text-to-Speech)  
10. Packaging for Demonstration and Reuse

## Technologies Used
- **Python**
- **Scikit-Learn**
- **NLTK**
- **Joblib**
- **Pandas / NumPy**
- **Matplotlib**
- **Streamlit**
- **gTTS / pyttsx3** (for text-to-speech functionality)

## Streamlit Application
The Streamlit application supports:
- User input for message classification
- Instant spam or ham prediction
- Optional auditory feedback via text-to-speech

### Run Locally
```bash
streamlit run app.py

