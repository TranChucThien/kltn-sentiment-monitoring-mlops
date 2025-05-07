import logging
import yaml
import os
from datetime import datetime
from utils.s3_process import read_csv_from_s3, push_csv_to_s3, upload_file_to_s3
from utils.clean_text import preprocess
from utils.s3_process import read_key



def main():
    # Setup logging
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = f'/tmp/job_{current_time}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.info("Logging setup complete")
    logger.info(f"Log file created at: {LOG_FILE}")
    
    try:
        logger.info("========== Job started ==========")
        logging.info("Initializing Spark session...")
        # Read config file
        logger.info("Loading configuration from 'configs/config.yaml'")
        logger.info("Loading secrets from 'configs/secrets.yaml'")
        logging.info("Loading configuration from 'configs/config.yaml'")
        logging.info("Loading secrets from 'configs/secrets.yaml'")
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        with open("configs/secrets.yaml", "r") as f:
            config_secret = yaml.safe_load(f)
        
        bucket = config['s3']['bucket']
        raw_key = config['s3']['keys']['raw_data']
        input_path = f"s3a://{bucket}/{raw_key}"
        logger.info(f"Input path for raw data: {input_path}")
        logging.info(f"Input path for raw data: {input_path}")
        
        AWS_KEY_PATH = config['aws']['access_key_path']
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
        AWS_REGION = config['aws']['region']
        S3_OUTPUT_KEY = config['s3']['keys']['dataset']
        BUCKET_NAME = config['s3']['bucket']
        
        # Read CSV file from S3 (raw data)
        logger.info("Reading CSV file from S3 (raw data)...")
        logging.info("Reading CSV file from S3 (raw data)...")
        df = read_csv_from_s3(input_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        row_count = df.count()
        logger.info(f"Successfully read raw data with {row_count} records")
        logging.info(f"Successfully read raw data with {row_count} records")
        df.show(3)
        
        # Preprocess data
        logger.info("Starting preprocessing data...")
        logging.info("Starting preprocessing data...")
        df = preprocess(df, input_path)
        processed_row_count = df.count()
        logger.info(f"Preprocessing completed. Number of records after preprocessing: {processed_row_count}")
        logging.info(f"Preprocessing completed. Number of records after preprocessing: {processed_row_count}")
        
        # Show the preprocessed data
        logger.info("Showing sample of preprocessed data:")
        logging.info("Showing sample of preprocessed data:")
        df.printSchema()
        df.show(3)
        
        # Push the preprocessed data to S3
        logger.info(f"Saving preprocessed data to S3 bucket '{BUCKET_NAME}' at key '{S3_OUTPUT_KEY}'")
        logging.info(f"Saving preprocessed data to S3 bucket '{BUCKET_NAME}' at key '{S3_OUTPUT_KEY}'")
        push_csv_to_s3(df, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME, S3_OUTPUT_KEY, LOCAL_TEMP_DIR='/tmp/spark_output')
        logger.info("Successfully pushed preprocessed data to S3")
        logging.info("Successfully pushed preprocessed data to S3")
        
        # Upload log file to S3
        logger.info("Uploading log file to S3")
        logging.info("Uploading log file to S3")
        log_s3_key = f"logs/job_{current_time}.log"
        upload_file_to_s3(LOG_FILE, BUCKET_NAME, log_s3_key, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        logger.info(f"Uploaded log file to S3 at: s3://{BUCKET_NAME}/{log_s3_key}")
        logging.info(f"Uploaded log file to S3 at: s3://{BUCKET_NAME}/{log_s3_key}")
        
        logger.info("========== Job completed successfully ==========")
        logging.info("========== Job completed successfully ==========")
        
    except Exception as e:
        logger.exception(f"Job failed due to error: {e}")
        raise
    finally:
        # Clean up local log file
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            logger.info("Removed local log file after upload")
    

if __name__ == "__main__":
    main()
