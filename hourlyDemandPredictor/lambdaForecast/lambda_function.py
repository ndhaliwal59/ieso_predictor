import json
import boto3
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import tempfile
from io import BytesIO
import warnings

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings('ignore')
pl.seed_everything(42, workers=True)

s3_client = boto3.client('s3')

MODEL_BUCKET = "energy-forecast-nishan"
MODEL_PREFIX = "models/current/"
OUTPUT_BUCKET = "energy-forecast-nishan"
OUTPUT_PREFIX = "daily_prediction/"

def download_from_s3(bucket, prefix, local_path):
    """Download file from S3 to local path"""
    response = s3_client.get_object(Bucket=bucket, Key=prefix)
    with open(local_path, 'wb') as f:
        f.write(response['Body'].read())

def upload_to_s3(local_path, bucket, key):
    """Upload file to S3"""
    s3_client.upload_file(local_path, bucket, key)

def load_historical_data():
    """
    Load and preprocess historical data from S3.
    Assumes your CSV is stored in S3 - adjust bucket/key as needed.
    """
    csv_bucket = "energy-forecast-nishan"
    csv_key = "training_dataset/combined_demand_2002_2025.csv"
    
    try:
        response = s3_client.get_object(Bucket=csv_bucket, Key=csv_key)
        df = pd.read_csv(BytesIO(response['Body'].read()))
    except Exception as e:
        raise Exception(f"Failed to load historical data from S3: {e}")
    
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=["Market Demand"], errors="ignore")
    
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Hour"])
    df["Hour"] = df["Hour"].astype(int).clip(1, 24)
    
    df["time"] = df["Date"] + pd.to_timedelta(df["Hour"] - 1, unit="h")
    df["Ontario Demand"] = pd.to_numeric(df["Ontario Demand"], errors="coerce")
    df = df.dropna(subset=["Ontario Demand", "time"]).sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"], keep="first").sort_values("time")
    
    full_range = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full_range).rename_axis("time").reset_index()
    df["Ontario Demand"] = df["Ontario Demand"].astype(float).ffill()
    
    df["series"] = "ON"
    df["time_idx"] = ((df["time"] - df["time"].min()).dt.total_seconds() // 3600).astype(int)
    df["hour"] = df["time"].dt.hour.astype("int16")
    df["day_of_week"] = df["time"].dt.dayofweek.astype("int8")
    df["month"] = df["time"].dt.month.astype("int8")
    
    return df

def lambda_handler(event, context):
    """AWS Lambda handler for daily Ontario demand forecasting"""
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download model from S3
            model_path = os.path.join(tmpdir, "tft_best_model.ckpt")
            download_from_s3(MODEL_BUCKET, f"{MODEL_PREFIX}tft_best_model.ckpt", model_path)
            
            # Load historical data
            df = load_historical_data()
            
            # Create reference training dataset
            max_encoder_length = 168
            max_prediction_length = 24
            
            training = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="Ontario Demand",
                group_ids=["series"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                time_varying_known_reals=["time_idx", "hour", "day_of_week", "month"],
                time_varying_unknown_reals=["Ontario Demand"],
                target_normalizer=GroupNormalizer(groups=["series"]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )
            
            # Prepare prediction data
            last_time = df["time"].max()
            last_time_idx = df["time_idx"].max()
            last_demand = df["Ontario Demand"].iloc[-1]
            
            future_times = pd.date_range(
                start=last_time + pd.Timedelta(hours=1),
                periods=max_prediction_length,
                freq="h"
            )
            
            future_rows = pd.DataFrame({
                "time": future_times,
                "series": "ON",
                "Ontario Demand": last_demand,
                "hour": future_times.hour.astype("int16"),
                "day_of_week": future_times.dayofweek.astype("int8"),
                "month": future_times.month.astype("int8"),
            })
            
            future_rows["time_idx"] = ((future_rows["time"] - df["time"].min()).dt.total_seconds() // 3600).astype(int)
            
            prediction_df = pd.concat([df, future_rows], ignore_index=True)
            
            # Create prediction dataset
            prediction_dataset = TimeSeriesDataSet.from_dataset(
                training,
                prediction_df,
                predict=True,
                stop_randomization=True
            )
            
            predict_loader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=0
            )
            
            # Load model
            tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
            tft.eval()
            
            # Generate predictions
            with torch.no_grad():
                raw_predictions = tft.predict(
                    predict_loader,
                    mode="prediction",
                    return_x=False,
                    return_index=False
                )
            
            # Extract predictions
            if isinstance(raw_predictions, torch.Tensor):
                predictions = raw_predictions.detach().cpu().numpy()
            else:
                predictions = raw_predictions
            
            if predictions.ndim == 3:
                predictions = predictions[0, :, 0]
            elif predictions.ndim == 2:
                predictions = predictions[0, :]
            else:
                predictions = predictions.flatten()
            
            predictions = predictions[:max_prediction_length]
            
            # Create output dataframe
            output_df = pd.DataFrame({
                "time": future_times[:len(predictions)],
                "predicted_ontario_demand": predictions,
                "horizon_hour_ahead": np.arange(1, len(predictions) + 1),
                "hour_of_day": future_times[:len(predictions)].hour,
                "day_of_week": future_times[:len(predictions)].dayofweek,
            })
            
            output_df['day_part'] = output_df['hour_of_day'].apply(
                lambda h: 'Night' if h < 6 or h >= 22 else 'Morning' if h < 12 else 'Afternoon' if h < 18 else 'Evening'
            )
            
            # Save to S3
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_key = f"{OUTPUT_PREFIX}forecast_{timestamp}.csv"
            
            output_csv_path = os.path.join(tmpdir, "forecast.csv")
            output_df.to_csv(output_csv_path, index=False)
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, output_key)
            
            # Also save latest as current forecast
            latest_key = f"{OUTPUT_PREFIX}latest_forecast.csv"
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, latest_key)
            
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Forecast completed successfully",
                    "forecast_key": output_key,
                    "latest_key": latest_key,
                    "predictions_count": len(predictions),
                    "mean_demand": float(predictions.mean()),
                    "min_demand": float(predictions.min()),
                    "max_demand": float(predictions.max()),
                })
            }
            
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
