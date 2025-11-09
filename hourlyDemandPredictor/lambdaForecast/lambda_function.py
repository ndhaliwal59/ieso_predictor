import json
import boto3
import os
from datetime import datetime, timedelta

# ============================================================================
# CRITICAL: DISABLE CUDA COMPLETELY - MUST BE FIRST
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

import pandas as pd
import numpy as np
import torch

# Force CPU-only mode
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: None
torch.cuda.get_device_name = lambda x: None

if hasattr(torch._C, '_cuda_init'):
    torch._C._cuda_init = lambda: None
if hasattr(torch._C, '_cuda_getDeviceCount'):
    torch._C._cuda_getDeviceCount = lambda: 0
if hasattr(torch._C, '_cuda_isDriverSufficient'):
    torch._C._cuda_isDriverSufficient = lambda: False

torch.backends.cudnn.enabled = False

import tempfile
from io import BytesIO
import warnings
import logging

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')
pl.seed_everything(42, workers=True)

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')

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
    """Load and preprocess historical data from S3"""
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
    
    logger.info("=" * 80)
    logger.info("Starting Ontario demand forecast generation (CPU-ONLY MODE)")
    logger.info("=" * 80)
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Created temporary directory: {tmpdir}")
            
            # ================================================================
            # Step 1: Download model from S3
            # ================================================================
            logger.info("Step 1: Downloading model from S3")
            model_path = os.path.join(tmpdir, "tft_best_model.ckpt")
            download_from_s3(MODEL_BUCKET, f"{MODEL_PREFIX}tft_best_model.ckpt", model_path)
            
            # ================================================================
            # Step 2: Load historical data
            # ================================================================
            logger.info("Step 2: Loading historical data")
            df = load_historical_data()
            
            # ================================================================
            # Step 3: Create reference training dataset
            # ================================================================
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
            logger.info(f"TimeSeriesDataSet created with {len(training)} samples")

            # ================================================================
            # Step 4: Prepare prediction data
            # ================================================================
            logger.info("Step 4: Preparing prediction data")
            last_time = df["time"].max()
            last_time_idx = df["time_idx"].max()
            last_demand = df["Ontario Demand"].iloc[-1]
            
            logger.info(f"Last time in data: {last_time}")
            logger.info(f"Last demand: {last_demand:.1f} MW")
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
            
            # ================================================================
            # Step 5: Create prediction dataset
            # ================================================================
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
            logger.info(f"Prediction dataset created with {len(prediction_dataset)} samples")

            # ================================================================
            # Step 6: Load checkpoint MANUALLY (avoid .to() calls)
            # ================================================================
            logger.info("Step 6: Loading checkpoint manually (bypass GPU checks)")
            
            # Load checkpoint dict
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info("✓ Checkpoint loaded to CPU")
            
            state_dict = ckpt.get("state_dict", ckpt)
            hyperparams = ckpt.get("hyper_parameters", {}) or {}
            
            # CRITICAL: Extract and restore the fitted normalizer
            if 'hyper_parameters' in ckpt and 'target_normalizer' in ckpt['hyper_parameters']:
                saved_normalizer = ckpt['hyper_parameters']['target_normalizer']
                logger.info(f"✓ Found saved normalizer: {type(saved_normalizer).__name__}")
                
                # Move normalizer to CPU if it has device-dependent state
                if hasattr(saved_normalizer, 'to'):
                    saved_normalizer = saved_normalizer.to('cpu')
                
                # Replace training dataset's normalizer with the fitted one
                training.target_normalizer = saved_normalizer
                logger.info("✓ Replaced training normalizer with fitted checkpoint normalizer")
            else:
                logger.warning("⚠ No saved normalizer found in checkpoint!")
            
            # Remove conflicting keys
            conflicting_keys = ['dataset', 'max_encoder_length', 'max_prediction_length', 
                              'time_varying_known_categoricals', 'time_varying_known_reals', 
                              'time_varying_unknown_categoricals', 'time_varying_unknown_reals',
                              'static_categoricals', 'static_reals', 'device']
            for key in conflicting_keys:
                hyperparams.pop(key, None)
            
            # Handle loss
            if 'loss' in hyperparams and isinstance(hyperparams['loss'], dict) and 'quantiles' in hyperparams['loss']:
                quantiles = hyperparams['loss']['quantiles']
                hyperparams['loss'] = QuantileLoss(quantiles=quantiles)
                hyperparams['output_size'] = len(quantiles)
            else:
                standard_quantiles = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
                hyperparams['loss'] = QuantileLoss(quantiles=standard_quantiles)
                hyperparams['output_size'] = len(standard_quantiles)
            
            # Create model from dataset (with fitted normalizer)
            logger.info("Creating model from dataset...")
            tft = TemporalFusionTransformer.from_dataset(training, **hyperparams)
            
            # Load state dict (move all tensors to CPU)
            new_state = {}
            for k, v in state_dict.items():
                k_clean = k.replace("model.", "")
                if isinstance(v, torch.Tensor):
                    new_state[k_clean] = v.cpu()
                else:
                    new_state[k_clean] = v
            
            missing, unexpected = tft.load_state_dict(new_state, strict=False)
            logger.info(f"✓ State dict loaded: {len(missing)} missing, {len(unexpected)} unexpected")
            
            # CRITICAL: Disable ALL metrics to prevent GPU device checks
            if hasattr(tft, 'logging_metrics'):
                tft.logging_metrics = torch.nn.ModuleList([])
            if hasattr(tft, 'validation_metrics'):
                tft.validation_metrics = torch.nn.ModuleList([])
            if hasattr(tft, 'test_metrics'):
                tft.test_metrics = torch.nn.ModuleList([])
            logger.info("✓ All metrics disabled")
            
            # Set output transformer (should already be set from training dataset)
            tft.output_transformer = training.target_normalizer
            
            # Move to CPU and set eval mode
            tft = tft.to('cpu')
            tft.eval()
            logger.info("✓ Model on CPU and in eval mode")

            # ================================================================
            # Step 7: Manual prediction (no Trainer, no GPU calls)
            # ================================================================
            logger.info("Step 7: Generating predictions manually (CPU-only)")
            
            all_preds = []
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(predict_loader):
                    # Ensure CPU
                    if isinstance(x, dict):
                        x = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                    else:
                        x = x.cpu()
                    
                    # Forward pass
                    output = tft(x)
                    
                    # Extract prediction
                    if isinstance(output, dict) and "prediction" in output:
                        pred = output["prediction"]
                    elif isinstance(output, tuple):
                        pred = output[0]
                    else:
                        pred = output
                    
                    all_preds.append(pred.detach().cpu())
                    logger.info(f"Processed batch {batch_idx + 1}, output shape: {pred.shape}")
            
            # Concatenate predictions
            predictions = torch.cat(all_preds, dim=0)
            logger.info(f"Raw concatenated shape: {predictions.shape}")
            
            # Extract median quantile
            if predictions.ndim == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[0, :, median_idx]
                logger.info(f"Extracted median quantile (index {median_idx})")
            elif predictions.ndim == 2:
                predictions = predictions[0, :]
            else:
                predictions = predictions.flatten()
            
            predictions = predictions[:max_prediction_length].numpy()
            
            logger.info(f"✓ Predictions generated (shape: {predictions.shape})")
            logger.info(f"✓ Predictions ALREADY DENORMALIZED via output_transformer")
            logger.info(f"Prediction stats - Mean: {predictions.mean():.2f}, Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
            
            # ================================================================
            # Validation: Compare to historical scale
            # ================================================================
            historical_mean = df["Ontario Demand"].tail(168).mean()
            pred_mean = predictions.mean()
            pct_diff = abs(pred_mean - historical_mean) / historical_mean * 100
            
            logger.info(f"Scale validation:")
            logger.info(f"  Historical mean (last 7 days): {historical_mean:.1f} MW")
            logger.info(f"  Prediction mean: {pred_mean:.1f} MW")
            logger.info(f"  Difference: {abs(pred_mean - historical_mean):.1f} MW ({pct_diff:.1f}%)")
            
            if pct_diff > 15:
                logger.warning(f"⚠ Large difference ({pct_diff:.1f}%) - predictions may be incorrect!")
            else:
                logger.info(f"✓ Predictions within expected scale")

            # ================================================================
            # Step 8: Create output dataframe
            # ================================================================
            logger.info("Step 8: Creating output dataframe")
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
            
            # ================================================================
            # Step 9: Save to S3
            # ================================================================
            logger.info("Step 9: Saving forecast to S3")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_key = f"{OUTPUT_PREFIX}forecast_{timestamp}.csv"
            
            output_csv_path = os.path.join(tmpdir, "forecast.csv")
            output_df.to_csv(output_csv_path, index=False)
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, output_key)
            
            # Also save as latest forecast
            latest_key = f"{OUTPUT_PREFIX}latest_forecast.csv"
            upload_to_s3(output_csv_path, OUTPUT_BUCKET, latest_key)
            
            logger.info("=" * 80)
            logger.info("FORECAST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Forecast period: {future_times[0]} to {future_times[-1]}")
            logger.info(f"Mean demand: {pred_mean:.1f} MW")
            logger.info(f"Min demand: {predictions.min():.1f} MW")
            logger.info(f"Max demand: {predictions.max():.1f} MW")
            
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Forecast completed successfully",
                    "forecast_key": output_key,
                    "latest_key": latest_key,
                    "predictions_count": len(predictions),
                    "forecast_date": future_times[0].strftime("%Y-%m-%d"),
                    "mean_demand": float(predictions.mean()),
                    "min_demand": float(predictions.min()),
                    "max_demand": float(predictions.max()),
                    "historical_mean": float(historical_mean),
                    "percent_difference": float(pct_diff),
                })
            }
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error("FORECAST FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "error_type": type(e).__name__
            })
        }
