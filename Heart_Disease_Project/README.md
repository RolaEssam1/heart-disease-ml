# Heart Disease Prediction - Machine Learning Pipeline

## Overview
This project implements a **complete Machine Learning pipeline** using the **UCI Heart Disease dataset**.  
It covers:
- Data preprocessing & cleaning  
- Dimensionality reduction (PCA)  
- Feature selection  
- Supervised learning (Logistic Regression, Decision Tree, Random Forest, SVM)  
- Unsupervised learning (KMeans, Hierarchical Clustering)  
- Hyperparameter tuning  
- Model export & deployment with **Streamlit UI**  

The goal is to build an end-to-end workflow for predicting heart disease risk.

---

## Project Structure
Heart_Disease_Project/
 data/
    heart_disease.csv # raw UCI dataset (renamed from processed.cleveland.data)
    heart_disease_clean.csv # cleaned dataset
 notebooks/
    01_data_preprocessing.ipynb
    02_pca_analysis.ipynb
    03_feature_selection.ipynb
    04_supervised_learning.ipynb
    05_unsupervised_learning.ipynb
    06_hyperparameter_tuning.ipynb
 models/
    final_model.pkl # trained ML model
 results/
    pca_transformed.csv
    selected_features.csv
    supervised_results.csv
    evaluation_metrics.txt
 ui/
    app.py # Streamlit web app
requirements.txt
README.md
.getignore


## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/RolaEssam1/heart-disease-ml
   cd heart-disease-ml

2. Install dependencies:

    pip install -r requirements.txt
