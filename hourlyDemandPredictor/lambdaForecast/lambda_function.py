import json
import boto3
import os
from datetime import datetime, timedelta

# DISABLE CUDA IMMEDIATELY - BEFORE ALL OTHER IMPORTS
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
import numpy as np
import torch

# Force CPU-only mode - critical for Lambda
torch.cuda.is_available = lambda: False
torch.cuda.init = lambda: None
torch.backends.cudnn.enabled = False

# Prevent CUDA initialization entirely - CRITICAL FIX
if hasattr(torch._C, '_cuda_init'):
    torch._C._cuda_init = lambda: None
if hasattr(torch._C, '_cuda_getDeviceCount'):
    torch._C._cuda_getDeviceCount = lambda: 0

import tempfile
from io import BytesIO
import warnings
import logging

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Additional CUDA disabling
torch.cuda.is_available = lambda: False

warnings.filterwarnings('ignore')
pl.seed_everything(42, workers=True)

# Set PyTorch Lightning to CPU mode
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

s3_client = boto3.client('s3')

MODEL_BUCKET = "energy-forecast-nishan"
MODEL_PREFIX = "models/current/"
OUTPUT_BUCKET = "energy-forecast-nishan"
OUTPUT_PREFIX = "daily_prediction/"

def download_from_s3(bucket, prefix, local_path):
    """Download file from S3 to local path"""
    logger.info(f"Downloading s3://{bucket}/{prefix}")
    response = s3_client.get_object(Bucket=bucket, Key=prefix)
    with open(local_path, 'wb') as f:
        f.write(response['Body'].read())
    logger.info(f"Downloaded successfully to {local_path}")

def upload_to_s3(local_path, bucket, key):
    """Upload file to S3"""
    logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3_client.upload_file(local_path, bucket, key)
    logger.info(f"Upload successful")

def load_historical_data():
    """
    Load and preprocess historical data from S3.
    Assumes your CSV is stored in S3 - adjust bucket/key as needed.
    """
    csv_bucket = "energy-forecast-nishan"
    csv_key = "training_dataset/combined_demand_2002_2025.csv"
    
    try:
        logger.info(f"Loading historical data from s3://{csv_bucket}/{csv_key}")
        response = s3_client.get_object(Bucket=csv_bucket, Key=csv_key)
        df = pd.read_csv(BytesIO(response['Body'].read()))
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load historical data: {e}", exc_info=True)
        raise Exception(f"Failed to load historical data from S3: {e}")
    
    logger.info("Preprocessing data...")
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
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df

def lambda_handler(event, context):
    """AWS Lambda handler for daily Ontario demand forecasting"""
    
    logger.info("=" * 50)
    logger.info("Starting Ontario demand forecast generation")
    logger.info("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Created temporary directory: {tmpdir}")
            
            # Download model from S3
            logger.info("Step 1: Downloading model from S3")
            model_path = os.path.join(tmpdir, "tft_best_model.ckpt")
            download_from_s3(MODEL_BUCKET, f"{MODEL_PREFIX}tft_best_model.ckpt", model_path)
            
            # Load historical data
            logger.info("Step 2: Loading historical data")
            df = load_historical_data()
            
            # Create reference training dataset
            logger.info("Step 3: Creating TimeSeriesDataSet")
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
            logger.info("TimeSeriesDataSet created successfully")
            
            # Prepare prediction data
            # Step 4: Preparing prediction data for TODAY (current calendar day)
            logger.info("Step 4: Preparing prediction data for current day")

            # Get today's date (the day that just started at midnight)
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(hours=23)

            logger.info(f"Forecasting for: {today_start.date()}")
            logger.info(f"Current time: {now}")

            # Calculate the last data point
            last_time = df["time"].max()
            last_demand = df["Ontario Demand"].iloc[-1]

            logger.info(f"Last time in data: {last_time}")

            # Calculate hours from last data point to reach today's midnight (if needed)
            # then add 24 hours for the full day
            if last_time < today_start:
                # Need to fill gap to reach today's midnight
                hours_gap = int((today_start - last_time).total_seconds() / 3600)
                total_hours = hours_gap + max_prediction_length
                start_time = last_time + pd.Timedelta(hours=1)
            else:
                # Already at or past midnight, just predict remaining hours
                hours_gap = 0
                total_hours = max_prediction_length
                start_time = last_time + pd.Timedelta(hours=1)

            logger.info(f"Generating {total_hours} hours of predictions")

            # Generate future times
            future_times = pd.date_range(
                start=start_time,
                periods=total_hours,
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
            logger.info(f"Prediction dataframe shape: {prediction_df.shape}")

            
            # Create prediction dataset
            logger.info("Step 5: Creating prediction dataset")
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
            logger.info("Prediction dataloader created")
            
            # Load model
            logger.info("Step 6: Loading TFT model from checkpoint (CPU-safe manual load)")
            
            # load checkpoint with torch directly (CPU-mapped)
            ckpt = torch.load(model_path, map_location="cpu")
            logger.info("Checkpoint loaded with torch.load (map_location=cpu)")
            
            # extract the saved state_dict (Lightning saves under 'state_dict')
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            
            # remove potential device references inside hyper_parameters if present
            hyperparams = ckpt.get("hyper_parameters", {}) or {}
            if "device" in hyperparams:
                hyperparams.pop("device", None)
            
            # Build new model from saved hyperparameters
            model_kwargs = {}
            for k in ("learning_rate", "hidden_size", "attention_head_size",
                    "dropout", "hidden_continuous_size"):
                if k in hyperparams:
                    model_kwargs[k] = hyperparams[k]
            
            # Fallback defaults
            if "hidden_size" not in model_kwargs:
                model_kwargs["hidden_size"] = 32
            if "attention_head_size" not in model_kwargs:
                model_kwargs["attention_head_size"] = 4
            if "dropout" not in model_kwargs:
                model_kwargs["dropout"] = 0.1
            if "hidden_continuous_size" not in model_kwargs:
                model_kwargs["hidden_continuous_size"] = 16
            if "learning_rate" not in model_kwargs:
                model_kwargs["learning_rate"] = 1e-3
            
            logger.info(f"Attempting to instantiate TFT with params: {model_kwargs}")
            
            try:
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=model_kwargs["learning_rate"],
                    hidden_size=model_kwargs["hidden_size"],
                    attention_head_size=model_kwargs["attention_head_size"],
                    dropout=model_kwargs["dropout"],
                    hidden_continuous_size=model_kwargs["hidden_continuous_size"],
                )
                logger.info("TemporalFusionTransformer architecture instantiated (CPU)")
            except Exception as e:
                logger.error("Failed to instantiate TFT from_dataset, falling back to generic constructor", exc_info=True)
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    hidden_size=model_kwargs.get("hidden_size", 32),
                    attention_head_size=model_kwargs.get("attention_head_size", 4),
                    hidden_continuous_size=model_kwargs.get("hidden_continuous_size", 16),
                )
            
            # Normalize keys: remove leading "model." if present
            new_state = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith("model."):
                    new_key = k[len("model."):]
                new_state[new_key] = v
            
            # load into our freshly created tft
            missing, unexpected = tft.load_state_dict(new_state, strict=False)
            logger.info(f"Loaded state_dict into tft (missing keys: {len(missing)}, unexpected keys: {len(unexpected)})")
            
            # force CPU and eval
            tft = tft.cpu()
            tft.eval()
            logger.info("Model moved to CPU and set to eval()")
            
            # ===== MANUAL PREDICTION (AVOID tft.predict() which creates a Trainer) =====
            logger.info("Step 7: Generating predictions manually (no Trainer)")
            
            all_predictions = []
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(predict_loader):
                    # Move batch to CPU (should already be there, but ensure)
                    if isinstance(x, dict):
                        x = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                    else:
                        x = x.cpu()
                    
                    # Forward pass
                    output = tft(x)
                    
                    # Extract predictions - output structure depends on TFT configuration
                    if isinstance(output, dict) and "prediction" in output:
                        pred = output["prediction"]
                    elif isinstance(output, tuple):
                        pred = output[0]
                    else:
                        pred = output
                    
                    # Convert to numpy
                    pred_np = pred.detach().cpu().numpy()
                    all_predictions.append(pred_np)
                    
                    logger.info(f"Processed batch {batch_idx + 1}/{len(predict_loader)}")
            
            logger.info("Predictions generated")
            
            # Step 8: Extracting and filtering predictions for current day
            logger.info("Step 8: Extracting and reshaping predictions")

            # Concatenate all predictions
            if len(all_predictions) > 0:
                predictions = np.concatenate(all_predictions, axis=0)
            else:
                raise ValueError("No predictions generated")

            logger.info(f"Raw predictions shape after concatenation: {predictions.shape}")

            # Reshape if needed
            if predictions.ndim == 3:
                # Shape is typically (batch, time, features) - take last batch and first feature
                predictions = predictions[-1, :, 0]
                logger.info(f"Reshaped from 3D: extracted last batch, first feature")
            elif predictions.ndim == 2:
                # Shape is (batch, time) - take last batch
                predictions = predictions[-1, :]
                logger.info(f"Reshaped from 2D: extracted last batch")
            else:
                predictions = predictions.flatten()
                logger.info(f"Flattened predictions")

            logger.info(f"Predictions shape after reshaping: {predictions.shape}")
            logger.info(f"Predictions length: {len(predictions)}")
            logger.info(f"Future times length: {len(future_times)}")

            # *** CRITICAL FIX: Align predictions length with future_times ***
            min_length = min(len(future_times), len(predictions))
            future_times = future_times[:min_length]
            predictions = predictions[:min_length]

            logger.info(f"After alignment - future_times: {len(future_times)}, predictions: {len(predictions)}")
            logger.info(f"Prediction stats - Mean: {predictions.mean():.2f}, Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")

            # Filter predictions to only include TODAY'S 24 hours (00:00 to 23:00)
            logger.info("Filtering predictions for current calendar day only")

            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(hours=23)

            logger.info(f"Target forecast date: {today_start.date()}")
            logger.info(f"Filtering for hours between {today_start} and {today_end}")

            # Create mask for today's hours
            today_mask = (future_times >= today_start) & (future_times <= today_end)
            today_times = future_times[today_mask]
            today_predictions = predictions[today_mask]

            logger.info(f"Filtered to {len(today_predictions)} hours for today ({today_start.date()})")

            if len(today_predictions) != 24:
                logger.warning(f"WARNING: Expected 24 hours but got {len(today_predictions)}!")
                logger.warning(f"First prediction time: {future_times[0]}")
                logger.warning(f"Last prediction time: {future_times[-1]}")

            # Create output dataframe with today's predictions only
            output_df = pd.DataFrame({
                "time": today_times,
                "predicted_ontario_demand": today_predictions,
                "forecast_date": today_start.date(),
                "hour_of_day": today_times.hour,
                "day_of_week": today_times.dayofweek,
            })

            output_df['day_part'] = output_df['hour_of_day'].apply(
                lambda h: 'Night' if h < 6 or h >= 22 else 'Morning' if h < 12 else 'Afternoon' if h < 18 else 'Evening'
            )

            logger.info(f"Output dataframe shape: {output_df.shape}")
            logger.info(f"Output covers hours {output_df['hour_of_day'].min()} to {output_df['hour_of_day'].max()}")
            logger.info(f"Final prediction stats - Mean: {today_predictions.mean():.2f}, Min: {today_predictions.min():.2f}, Max: {today_predictions.max():.2f}")
            
            # Save to S3
            logger.info("Step 10: Saving forecast to S3")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_key = f"{OUTPUT_PREFIX}forecast_{timestamp}.csv"
            
            output_csv_path = os.path.join(tmpdir, "forecast.csv")
            output_df.to_csv(output_csv_path, index=False)
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, output_key)
            
            # Also save latest as current forecast
            latest_key = f"{OUTPUT_PREFIX}latest_forecast.csv"
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, latest_key)
            
            logger.info("=" * 50)
            logger.info("FORECAST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            
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
        logger.error("=" * 50)
        logger.error("FORECAST FAILED!")
        logger.error("=" * 50)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
