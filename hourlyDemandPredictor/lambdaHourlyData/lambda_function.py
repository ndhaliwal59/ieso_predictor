import os
import io
import json
import logging
from datetime import datetime, timezone, timedelta  # Added timedelta import
from zoneinfo import ZoneInfo
import boto3
import requests
import pandas as pd


# -------------------------------
# Configuration
# -------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)


S3_BUCKET = os.environ.get("S3_BUCKET", "energy-forecast-nishan")
S3_PREFIX = os.environ.get("S3_PREFIX", "hourly_data/")
MAIN_DATA_KEY = "training_dataset/combined_demand_2002_2025.csv"


s3 = boto3.client("s3")


IESO_URL = "https://www.ieso.ca/ieso/api/HomePageWebApi/getHomePageData"


# Toronto timezone (handles EST/EDT automatically)
TORONTO_TZ = ZoneInfo("America/Toronto")


# -------------------------------
# Helper Functions
# -------------------------------
def fetch_ieso():
    """Fetch data from the IESO API."""
    resp = requests.get(IESO_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()


def build_payload(raw):
    """Build the structured payload with metadata and derived values."""
    extracted = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "IESO",
        "source_url": IESO_URL,
        "data": {
            "SupplyHour": raw.get("SupplyHour"),
            "Nuclear": raw.get("Nuclear"),
            "Wind": raw.get("Wind"),
            "Hydro": raw.get("Hydro"),
            "Solar": raw.get("Solar"),
            "Gas": raw.get("Gas"),
            "Biofuel": raw.get("Biofuel"),
            "HourlyImports": raw.get("HourlyImports"),
            "HourlyExports": raw.get("HourlyExports"),
            "OntarioDemand": raw.get("OntarioDemand"),
            "OntarioDemandHour": raw.get("OntarioDemandHour"),
            "ProjectedDemand": raw.get("ProjectedDemand"),
            "ProjectedDemandHour": raw.get("ProjectedDemandHour"),
            "UpdatedAt": raw.get("UpdatedAt"),
            "ReportForDate": raw.get("ReportForDate")
        }
    }


    # Derived metrics
    try:
        total_supply = (
            (raw.get('Nuclear') or 0)
            + (raw.get('Wind') or 0)
            + (raw.get('Hydro') or 0)
            + (raw.get('Solar') or 0)
            + (raw.get('Gas') or 0)
            + (raw.get('Biofuel') or 0)
        )
        net_flow = (raw.get('HourlyImports') or 0) - (raw.get('HourlyExports') or 0)
        extracted["derived"] = {"total_supply": total_supply, "net_flow": net_flow}
    except Exception as e:
        logger.warning("Failed to compute derived fields: %s", e)


    return extracted


def s3_put_json(bucket, key, payload):
    """Upload JSON payload to S3."""
    body = json.dumps(payload, default=str)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Uploaded hourly JSON to s3://%s/%s", bucket, key)


def make_s3_key(prefix):
    """Generate a time-based S3 key (e.g., hourly_data/2025-11-03T15-00Z.json)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = f"{ts}.json"
    prefix = prefix if prefix.endswith("/") else prefix + "/"
    return f"{prefix}{filename}"


def append_to_main_dataset(bucket, new_row):
    """Append a new row to the main dataset CSV stored in S3."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=MAIN_DATA_KEY)
        existing = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        # Create new file if missing
        existing = pd.DataFrame(columns=["Date", "Hour", "Ontario Demand"])


    # Append new row
    updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)


    # Write back to S3
    with io.StringIO() as csv_buffer:
        updated.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=MAIN_DATA_KEY,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )


    logger.info("Main dataset updated with new row: %s", new_row)


# -------------------------------
# Main Lambda Handler
# -------------------------------
def lambda_handler(event, context):
    logger.info("Starting hourly fetch lambda")


    # Step 1: Fetch from API
    try:
        raw = fetch_ieso()
    except Exception as e:
        logger.exception("Error fetching from IESO: %s", e)
        raise


    # Step 2: Build structured payload
    payload = build_payload(raw)


    # Step 3: Upload the hourly JSON
    key = make_s3_key(S3_PREFIX)
    try:
        s3_put_json(S3_BUCKET, key, payload)
    except Exception as e:
        logger.exception("Error uploading hourly JSON to S3: %s", e)
        raise


    # Step 4: Append new row to training dataset
    try:
        # Convert UTC to Toronto local time
        utc_now = datetime.now(timezone.utc)
        toronto_now = utc_now.astimezone(TORONTO_TZ)
        
        # Calculate minutes past midnight for window check
        minutes_past_midnight = toronto_now.hour * 60 + toronto_now.minute
        
        if 7 <= minutes_past_midnight <= 67:  # 00:07 to 01:07 Toronto time
            # Adjust for previous day: Hour 24
            previous_date = toronto_now.date() - timedelta(days=1)
            local_date = previous_date.strftime("%Y-%m-%d")
            ontario_demand = payload["data"].get("OntarioDemand", 0) - 750
            new_row = {
                "Date": local_date,
                "Hour": 24,
                "Ontario Demand": ontario_demand
            }
            logger.info("Adjusted for midnight rollover: previous date %s, hour 24", local_date)
        else:
            # Original logic
            local_date = toronto_now.strftime("%Y-%m-%d")
            new_row = {
                "Date": local_date,
                "Hour": payload["data"].get("OntarioDemandHour"),
                "Ontario Demand": payload["data"].get("OntarioDemand")
            }
            logger.info("Standard row for date: %s, hour: %s", local_date, new_row["Hour"])
        
        append_to_main_dataset(S3_BUCKET, new_row)
    except Exception as e:
        logger.exception("Error appending new row to dataset: %s", e)
        raise


    logger.info("Fetch complete — JSON and CSV updated.")
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "success",
            "s3_json_key": key,
            "dataset_key": MAIN_DATA_KEY
        })
    }
