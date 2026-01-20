# Accessible Spam Message Classifier

A machine learning–based [SMS spam detection application](https://spammsgcheck.streamlit.app/) that classifies messages as spam or ham (not spam) using **Natural Language Processing (NLP)** and a **Support Vector Machine (SVM)** model, deployed via Streamlit with text-to-speech output for accessibility.

*Please click [here](https://youtu.be/EwgdOBuiui4) for video demo.*

<img src="./images/app.png" width="" height="500">

## Highlights
- End-to-end workflow from preprocessing to deployment.
- TF-IDF vectorization for text feature extraction.
- SVM classifier optimized and stratifying train-test split for imbalanced data.
- Exported model and vectorizer for reproducible inference (joblib).
- [Streamlit application](https://spammsgcheck.streamlit.app/) with accessible user interaction and audio output.

## Skills Demonstrated
✔ NLP and Feature Engineering

✔ Supervised Machine Learning (SVM)

✔ Model Evaluation and Metric Selection

✔ Imbalanced Classification Strategies

✔ Model Serialization and Deployment

✔ Streamlit Application Development

✔ Accessibility-Aware UX Design (Text-to-Speech)

## Problem Statement
Spam messages can pose significant risks such as phishing, scams, and misinformation, which disproportionately affect vulnerable groups. This [application](https://spammsgcheck.streamlit.app/) aims not only to detect spam accurately but also to present the results in a clear, accessible way — including the option for the outcome to be read aloud, ensuring that users with limited vision or reading ability can easily understand the classification result.

## Project Overview

> **Note:**  
> *This Streamlit application is hosted on the free Tier of Streamlit Community Cloud. If the app has been idle for more than 12 hours, it may take some time to reactivate. In such cases, please click the button saying “Yes, get this app back up!” to relaunch the application. Thank you for your patience.*

This project implements a complete machine learning workflow including:
1. Dataset acquisition ([SMS spam dataset from Kaggle](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fuciml%2Fsms-spam-collection-dataset%3Fresource%3Ddownload))
2. Text normalization and preprocessing
3. Exploratory data analysis
4. TF-IDF vectorization for numeric representation
5. SVM model training with hyperparameter tuning
6. F1-score optimization due to class imbalance
7. Export of trained classifier and vectorizer
8. Streamlit deployment with auditory feedback

#### Class Imbalance of the Dataset:

<img src="./images/data.png" width="" height="500">


## Key Technical Decisions
The following design choices were made to address data imbalance, optimize feature representation, and ensure deployability:
- Chose TF-IDF over count vectors due to sparse representation advantages
- Selected SVM as classifier due to better margin performance on sparse text data
- Used stratified train-test split and F1 score (balancing precision and recall for the minority spam class) to address class imbalance


## Key Insights & Impacts
- Reduces user exposure to fraudulent SMS content by providing fast and automated classification.
- Text-to-speech functionality improves accessibility for populations such as older adults and individuals with visual impairments.
- Provides instant and interpretable feedback, increasing user awareness of spam indicators and digital communication safety.
- Guidance on how to handle suspicious text messages, with links to the relevant government resources.


## Results Summary

The spam detection model was evaluated on a held-out test dataset.

**Model Configuration**
- Classifier: Support Vector Machine (SVM)
- Feature Extraction: TF-IDF

**Performance Metrics**
- Primary F1 Score: 0.92 (test set)

**Confusion Matrix**
|               | Predicted: Ham | Predicted: Spam |
|---------------|----------------|-----------------|
| Actual: Ham   | 897            | 7               |
| Actual: Spam  | 16             | 133             |

**Classification Report (Weighted Averages)**
- Precision: 0.98
- Recall: 0.98
- F1 Score: 0.98

> *Notes:
> Results reflect evaluation on a static [dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fuciml%2Fsms-spam-collection-dataset%3Fresource%3Ddownload)). Real-world performance may vary depending on the nature of user-provided input.*


## Stages of Development
1. Data Loading and Inspection  
2. Text Preprocessing and Normalization  
3. Exploratory Data Analysis  
4. TF-IDF Feature Engineering  
5. Model Training and Hyperparameter Tuning  
6. Model Validation and Metric Comparison  
7. Model Serialization
8. Streamlit Application Development  
9. Accessibility Enhancement (Text-to-Speech)  
10. Packaging for Demonstration and Reuse

## Technologies Used
- **Scikit-Learn**
- **NLTK**
- **Joblib**
- **Pandas**
- **Matplotlib**
- **Streamlit**

## Author
Carmen Wong
