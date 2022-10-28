
import boto3
import logging
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

# Define data source and other parameters.
SRC_BUCKET = 'prod-dcd-datasets-cache-zipfiles'
SRC_KEY = 'zr7vgbcyr2-1.zip'
DATA_DIR = '/opt/ml/processing/input'

# Download raw data zip from https://data.mendeley.com/datasets/zr7vgbcyr2/1
logging.info(f'Downloading {SRC_KEY}')
s3_boto_client = boto3.client("s3")
s3_boto_client.download_file(SRC_BUCKET, SRC_KEY, f'{DATA_DIR}/raw.zip')

# Unzip data
logging.info(f'Unpacking {SRC_KEY}')
shutil.unpack_archive(f'{DATA_DIR}/raw.zip', DATA_DIR)
for i in range(1,4):    
    logging.info(f'Unpacking imgs_part_{i}.zip')
    shutil.unpack_archive(f'{DATA_DIR}/images/imgs_part_{i}.zip', f'{DATA_DIR}/images')
    logging.info(f'Copying {DATA_DIR}/images/imgs_part_{i} to {DATA_DIR}/images/all_imgs')
    shutil.copytree(f'{DATA_DIR}/images/imgs_part_{i}', f'{DATA_DIR}/images/all_imgs', dirs_exist_ok=True)

# Split data into training, validation, and test sets
logging.info(f'Creating training-validation data split')
metadata = pd.read_csv(f'{DATA_DIR}/metadata.csv')
train_df, test_df = train_test_split(metadata, test_size=0.2, stratify=metadata['diagnostic'])
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['diagnostic'])

# Copy training data into folders for training
logging.info(f'Copying training data to {DATA_DIR}/images/output/train')
os.makedirs(f"{DATA_DIR}/output/train", exist_ok=True)
train_df.to_csv(f'{DATA_DIR}/output/train/metadata.csv', index=False)
for _,row in train_df.iterrows():
    src = f"{DATA_DIR}/images/all_imgs/{row['img_id']}"
    os.makedirs(f"{DATA_DIR}/output/train/{row['diagnostic']}", exist_ok=True)
    dest = f"{DATA_DIR}/output/train/{row['diagnostic']}/{row['img_id']}"
    shutil.copy2(src, dest)   
    
# Copy validation data into folders for training
logging.info(f'Copying validation data to {DATA_DIR}/images/output/val')
os.makedirs(f"{DATA_DIR}/output/val", exist_ok=True)
train_df.to_csv(f'{DATA_DIR}/output/val/metadata.csv', index=False)
for _,row in val_df.iterrows():
    src = f"{DATA_DIR}/images/all_imgs/{row['img_id']}"
    os.makedirs(f"{DATA_DIR}/output/val/{row['diagnostic']}", exist_ok=True)
    dest = f"{DATA_DIR}/output/val/{row['diagnostic']}/{row['img_id']}"
    shutil.copy2(src, dest)
    
# Copy test data into folders for evaluation
logging.info(f'Copying test data to {DATA_DIR}/images/output/test')
os.makedirs(f"{DATA_DIR}/output/test", exist_ok=True)
train_df.to_csv(f'{DATA_DIR}/output/test/metadata.csv', index=False)
for _,row in val_df.iterrows():
    src = f"{DATA_DIR}/images/all_imgs/{row['img_id']}"
    os.makedirs(f"{DATA_DIR}/output/test/{row['diagnostic']}", exist_ok=True)
    dest = f"{DATA_DIR}/output/test/{row['diagnostic']}/{row['img_id']}"
    shutil.copy2(src, dest)
