# ML-Price-Prediction-Pipeline

## Overview
This repository contains an end-to-end Machine Learning solution for predicting the sale price of heavy equipment. Using historical auction data, the project implements a robust regression pipeline to forecast prices based on equipment attributes and time-series data, Based on the [Kaggle Blue Book for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers).

## Key Features
* **Automated Pipeline:** Integrated preprocessing, feature engineering, and training via Scikit-Learn Pipelines.
* **Fast Inference:** Includes a dedicated prediction script to test the model on samples without retraining.
* **Production Standards:** Professional logging, dynamic path management, and structured error handling.
* **Feature Engineering:** Custom transformers to extract seasonal and temporal features from sales dates.
* **Scalability:** Designed with modular functions to allow easy integration of new data or models.

## Project Structure
```text
├── data/               # Project data (Raw Kaggle CSVs)
├── models/             # Directory for trained model output (Created automatically)
├── notebooks/          # Exploratory Data Analysis and research
├── src/                # Production-grade source code
│   ├── pipeline.py     # Main training and evaluation script
│   └── predict_samples.py # Script for generating predictions on samples
├── venv/               # Virtual environment (Not uploaded to Git)
├── requirements.txt    # Project dependencies (Pinned versions) 
└── README.md           # Project documentation
```
## Setup & Installation

1. **Prerequisites**
* **Python 3.8+** (Developed with 3.13)
* Kaggle account to download the competition data
2. **Clone the repository:**
   ```bash
   git clone https://github.com/lotantamary/ML-Price-Prediction-Pipeline.git
   cd ML-Price-Prediction-Pipeline
   ```
3. **Environment Setup:**
   ```bash
   python -m venv venv # Create virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies:**
* The requirements.txt file contains all necessary libraries including Scikit-Learn, Pandas, and Joblib:
   ```bash
   pip install -r requirements.txt
   ```
### 5. Data Acquisition & Extraction
1. **Download:** Get the dataset from [Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data).
2. **Structure:** Create the following directory path: `data/bluebook-for-bulldozers/`.
3. **Extract:** Place `Train.csv`, `Valid.csv`, and `ValidSolution.csv` inside that subfolder.

## Usage

### 1. Research & Model Development

To explore the data analysis, feature engineering, and the model training process, you can launch the interactive notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lotantamary/ML-Price-Prediction-Pipeline/blob/main/notebooks/research_and_eda.ipynb)

*The notebook includes Data Exploration (EDA), model architecture, and performance evaluation.*

### 2. Full Training Pipeline
* The training pipeline must be executed first to establish the predictive model.
To execute the full training and evaluation pipeline, run the following command from the project root:
   ```bash
   python src/pipeline.py
   ```
* The script will process the data, train the model, log performance metrics to the console, and save the final model to the models/ directory.


### 3. Model Inference & Validation Samples
* Once the pipeline has successfully serialized the model, you can perform inference on unseen data. This step provides a practical demonstration of the model's predictive capabilities by comparing its outputs against ground-truth values from the validation set.

* Run the following command from the project root:
   ```bash
   python src/predict_samples.py
   ```
* This script will load 10 random samples from the validation set and display the predicted vs. actual prices.


## Model Performance
The model is evaluated using Root Mean Squared Log Error (RMSLE) and Mean Absolute Error (MAE).
* **Validation MAE:** 5937.267
* **Validation RMSLE:** 0.246
* **Validation R²:** 0.882

## Tech Stack
* **Language:** Python
* **ML Framework:** Scikit-Learn (Random Forest, Pipelines)
* **Data Tools:** Pandas, NumPy
**Visualization:** Matplotlib, Seaborn
* **Deployment Tools:** Joblib (Serialization), Logging