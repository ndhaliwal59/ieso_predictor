# IESO Demand Predictor

A machine learning project for predicting Ontario electricity demand using Temporal Fusion Transformer (TFT) models.

## Overview

This project implements time series forecasting models to predict hourly Ontario electricity demand. The models are trained using historical demand data and deployed via AWS Lambda functions.

## Model Training

The model training code is available on Kaggle:

- **Baseline Model**: [IESO Baseline](https://www.kaggle.com/code/nishandhaliwal06/leso-baseline)
- **Prediction Model**: [IESO Predicting](https://www.kaggle.com/code/nishandhaliwal06/iesso-predicting)

## Project Structure

- `baselineModel.ipynb` - Baseline TFT model training
- `predictor.ipynb` - Model inference and prediction code
- `engineeringFeatures.ipynb` - Feature engineering pipeline
- `trainingFeatures.ipynb` - Training feature preparation
- `lambdaForecast/` - AWS Lambda function for forecasting
- `lambdaHourlyData/` - AWS Lambda function for hourly data processing
- `sagemaker/` - SageMaker training scripts
- `engineeredDatasets/` - Processed feature datasets
- `rawCSV/` - Raw historical data files

## Key Technologies

- PyTorch Lightning
- PyTorch Forecasting (Temporal Fusion Transformer)
- AWS Lambda
- AWS SageMaker

