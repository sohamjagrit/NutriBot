"""AWS S3 utilities for document upload/download."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class S3Manager:
    """Manages S3 operations for data ingestion."""

    def __init__(self, bucket_name: str, region: str, access_key: str = "", secret_key: str = ""):
        """Initialize S3 manager.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            access_key: AWS access key (uses env var if empty)
            secret_key: AWS secret key (uses env var if empty)
        """
        self.bucket_name = bucket_name
        self.region = region

        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")

        if access_key and secret_key:
            self.s3_client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        else:
            self.s3_client = boto3.client("s3", region_name=region)

        logger.info(f"S3Manager initialized for bucket '{bucket_name}' in region '{region}'")

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a local file to S3.

        Args:
            local_path: Path to local file
            s3_key: S3 object key (path)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3.

        Args:
            s3_key: S3 object key (path)
            local_path: Path where to save file locally

        Returns:
            True if successful, False otherwise
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in S3 with given prefix.

        Args:
            prefix: S3 prefix (folder path)
            max_keys: Maximum number of objects to return

        Returns:
            List of object metadata dicts
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            objects = response.get("Contents", [])
            logger.info(f"Listed {len(objects)} objects in {prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Failed to list objects with prefix '{prefix}': {e}")
            return []

    def upload_json(self, data: Dict[str, Any], s3_key: str) -> bool:
        """Upload a JSON object to S3.

        Args:
            data: Dictionary to upload as JSON
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            json_str = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str,
                ContentType="application/json",
            )
            logger.info(f"Uploaded JSON to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload JSON to {s3_key}: {e}")
            return False

    def download_json(self, s3_key: str) -> Dict[str, Any] | None:
        """Download a JSON object from S3.

        Args:
            s3_key: S3 object key

        Returns:
            Parsed JSON dict, or None if failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = json.loads(response["Body"].read())
            logger.info(f"Downloaded JSON from s3://{self.bucket_name}/{s3_key}")
            return data
        except ClientError as e:
            logger.error(f"Failed to download JSON from {s3_key}: {e}")
            return None

    def create_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist.

        Returns:
            True if bucket exists or was created, False otherwise
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' already exists")
            return True
        except ClientError:
            try:
                if self.region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                logger.info(f"Created bucket '{self.bucket_name}'")
                return True
            except ClientError as e:
                logger.error(f"Failed to create bucket: {e}")
                return False
