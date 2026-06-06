import boto3
import os

s3 = boto3.client('s3', endpoint_url=os.environ.get('MLFLOW_S3_ENDPOINT_URL'))
try:
    s3.create_bucket(Bucket='reviewsentinel-artifacts')
    print("✅ Bucket 'reviewsentinel-artifacts' created successfully!")
except Exception as e:
    print(f"Notice: {e}")
