# train.py - SageMaker TFT Training Script
import os
import argparse
from datetime import datetime
import boto3
import pandas as pd
import numpy as np
import torch

import lightning as L  # CHANGED: pytorch_lightning → lightning for 2.x
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


s3_client = boto3.client('s3')


def download_from_s3(s3_uri, local_path):
    """Download file from S3 to local path"""
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    s3_client.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")


def upload_to_s3(local_path, s3_uri):
    """Upload file from local path to S3"""
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    s3_client.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} to {s3_uri}")


def backup_current_model(bucket, current_key, versions_prefix):
    """Backup current model to versions folder with timestamp"""
    try:
        s3_client.head_object(Bucket=bucket, Key=current_key)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_key = f"{versions_prefix}tft_model_{timestamp}.ckpt"
        copy_source = {'Bucket': bucket, 'Key': current_key}
        s3_client.copy(copy_source, bucket, backup_key)
        print(f"Backed up current model to s3://{bucket}/{backup_key}")
    except s3_client.exceptions.NoSuchKey:
        print("No current model found to backup (first training)")
    except Exception as e:
        print(f"Could not backup model: {e}")


def load_and_preprocess_data(data_path):
    """Load and preprocess the training data"""
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    
    assert "Ontario Demand" in df.columns, "Expected 'Ontario Demand' column"
    
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Hour"]).copy()
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


def train_model(args):
    """Main training function"""
    L.seed_everything(42, workers=True)  # CHANGED: pl → L
    
    # Check if data is already downloaded by SageMaker
    # SageMaker downloads training data to /opt/ml/input/data/training/
    sagemaker_data_path = '/opt/ml/input/data/training/main_training_data.csv'
    local_data_path = '/tmp/main_training_data.csv'
    
    if os.path.exists(sagemaker_data_path):
        print(f"Using SageMaker-provided data from {sagemaker_data_path}")
        data_path = sagemaker_data_path
    else:
        print(f"Downloading training data from {args.data_s3_uri}")
        download_from_s3(args.data_s3_uri, local_data_path)
        data_path = local_data_path
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    print(f"Loaded {len(df)} rows of training data")
    
    max_encoder_length = 168
    max_prediction_length = 24
    
    cutoff = df["time_idx"].max() - max_prediction_length * 7
    training_df = df[df["time_idx"] <= cutoff].copy()
    validation_df = df[df["time_idx"] > cutoff].copy()
    
    print(f"Training samples: {len(training_df)}, Validation samples: {len(validation_df)}")
    
    # SAME NORMALIZER AND DATASET LOGIC
    training = TimeSeriesDataSet(
        training_df,
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
    
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    
    batch_size = args.batch_size
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, persistent_workers=False
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0, persistent_workers=False
    )
    
    # SAME MODEL INITIALIZATION
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        loss=QuantileLoss(),
        optimizer="adam",
        reduce_on_plateau_patience=3,
    )
    
    # SAME CALLBACKS
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="tft_best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=args.early_stop_patience, 
        mode="min", 
        verbose=True
    )
    
    # TRAINER WITH PYTORCH-LIGHTNING 2.1.0 COMPATIBILITY
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=None,  # CHANGED: False → None for 2.x
    )
    
    print("Starting training...")
    trainer.fit(tft, train_loader, val_loader)
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # SAME VALIDATION METRICS LOGIC
    loaded_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
    loaded_tft.eval()
    
    with torch.no_grad():
        preds = loaded_tft.predict(val_loader)
    
    actuals_list = []
    for _, y in val_loader:
        if isinstance(y, (list, tuple)):
            target = y[0]
        else:
            target = y
        if isinstance(target, list):
            target = target[0]
        actuals_list.append(target)
    
    actuals = torch.cat([t.detach().cpu().float() for t in actuals_list], dim=0)
    preds = preds.detach().cpu().float()
    
    mae = torch.mean(torch.abs(actuals - preds)).item()
    rmse = torch.sqrt(torch.mean((actuals - preds) ** 2)).item()
    smape = torch.mean(200.0 * torch.abs(actuals - preds) / 
                       (torch.abs(actuals) + torch.abs(preds) + 1e-6)).item()
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  sMAPE: {smape:.2f}%")
    
    # SAME METRICS SAVING
    os.makedirs(args.output_data_dir, exist_ok=True)
    metrics_df = pd.DataFrame({
        'metric': ['MAE', 'RMSE', 'sMAPE'],
        'value': [mae, rmse, smape],
        'timestamp': [datetime.utcnow().isoformat()] * 3
    })
    metrics_path = os.path.join(args.output_data_dir, 'validation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--attention-head-size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden-continuous-size', type=int, default=16)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    
    # S3 paths (used when data not provided by SageMaker)
    parser.add_argument('--data-s3-uri', type=str, 
                       default='s3://energy-forecast-nishan/training_dataset/combined_demand_2002_2025.csv')
    parser.add_argument('--model-s3-uri', type=str,
                       default='s3://energy-forecast-nishan/models/current/tft_best_model.ckpt')
    parser.add_argument('--versions-s3-prefix', type=str,
                       default='s3://energy-forecast-nishan/models/versions/')
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, 
                       default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    args = parser.parse_args()
    
    print("="*60)
    print("TFT Energy Demand Forecasting - Monthly Retraining")
    print("="*60)
    print(f"Model output directory: {args.model_dir}")
    print(f"Data output directory: {args.output_data_dir}")
    
    bucket = args.model_s3_uri.split('/')[2]
    current_model_key = '/'.join(args.model_s3_uri.split('/')[3:])
    versions_prefix = '/'.join(args.versions_s3_prefix.split('/')[3:])
    
    print("\nBacking up current model (if exists)...")
    backup_current_model(bucket, current_model_key, versions_prefix)
    
    best_model_path = train_model(args)
    
    print(f"\nUploading new model to {args.model_s3_uri}")
    upload_to_s3(best_model_path, args.model_s3_uri)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
