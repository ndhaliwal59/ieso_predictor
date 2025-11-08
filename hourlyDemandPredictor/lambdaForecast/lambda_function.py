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
            logger.info("Step 4: Preparing prediction data")
            last_time = df["time"].max()
            last_time_idx = df["time_idx"].max()
            last_demand = df["Ontario Demand"].iloc[-1]
            
            logger.info(f"Last time in data: {last_time}")
            logger.info(f"Forecasting next {max_prediction_length} hours")
            
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
            
            # Extract predictions
            logger.info("Step 8: Extracting and reshaping predictions")
            
            # Concatenate all predictions
            if len(all_predictions) > 0:
                predictions = np.concatenate(all_predictions, axis=0)
            else:
                raise ValueError("No predictions generated")
            
            # Reshape if needed
            if predictions.ndim == 3:
                # Shape is typically (batch, time, features) - take last batch and first feature
                predictions = predictions[-1, :, 0]
            elif predictions.ndim == 2:
                # Shape is (batch, time) - take last batch
                predictions = predictions[-1, :]
            else:
                predictions = predictions.flatten()
            
            predictions = predictions[:max_prediction_length]
            logger.info(f"Final predictions shape: {predictions.shape}")
            logger.info(f"Prediction stats - Mean: {predictions.mean():.2f}, Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
            
            # Create output dataframe
            logger.info("Step 9: Creating output dataframe")
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
            logger.info(f"Output dataframe shape: {output_df.shape}")
            
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