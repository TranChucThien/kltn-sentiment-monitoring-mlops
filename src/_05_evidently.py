import logging
import yaml
import os
from datetime import datetime
from utils.s3_process import read_csv_from_s3, push_csv_to_s3, upload_file_to_s3
from utils.clean_text import preprocess
from utils.s3_process import read_key
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from evidently import Dataset
from evidently import DataDefinition
from evidently.legacy.pipeline.column_mapping import ColumnMapping
import pandas as pd
from evidently import MulticlassClassification
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
from datetime import datetime


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
    logging.info("Logging setup complete")

    
    try:

        logging.info("Initializing Spark session...")
        # Read config file
        logging.info("Loading configuration from 'configs/config.yaml'")
        logging.info("Loading secrets from 'configs/secrets.yaml'")
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        with open("configs/secrets.yaml", "r") as f:
            config_secret = yaml.safe_load(f)
        
        bucket = config['s3']['bucket']
        test_result_key = config['s3']['keys']['test_result']
        test_result_new_key = config['s3']['keys']['test_result_new']
        
        test_result_path = f"s3a://{bucket}/{test_result_key}"
        test_result_new_path = f"s3a://{bucket}/{test_result_new_key}"
        
        logging.info(f"Input path for raw data: {test_result_path}")
        logging.info(f"Input path for new test result data: {test_result_new_path}")
        
        AWS_KEY_PATH = config['aws']['access_key_path']
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
        AWS_REGION = config['aws']['region']
        BUCKET_NAME = config['s3']['bucket']
        
        # Read CSV file from S3 (raw data)
        logging.info("Reading CSV file from S3 (raw data)...")
        spark_df_result = read_csv_from_s3(test_result_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        spark_df_result_new = read_csv_from_s3(test_result_new_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        
        row_count = spark_df_result.count()
        logging.info(f"Successfully read test result with {row_count} records")
        logging.info(f"Successfully read new test result data with {spark_df_result_new.count()} records")
        spark_df_result.show(3)
        spark_df_result_new.show(3)
   

        # Convert columns to StringType
        spark_df_result = spark_df_result.withColumn("label", col("label").cast(StringType()))
        spark_df_result = spark_df_result.withColumn("prediction", col("prediction").cast(StringType()))

        spark_df_result_new = spark_df_result_new.withColumn("label", col("label").cast(StringType()))
        spark_df_result_new = spark_df_result_new.withColumn("prediction", col("prediction").cast(StringType()))

        
        df_result = spark_df_result.select("text", "label", "prediction").toPandas()
        df_result['label'] = df_result['label'].astype(float)
        df_result['label'] = df_result['label'].astype(int)
        df_result['prediction'] = df_result['prediction'].astype(float)
        df_result['prediction'] = df_result['prediction'].astype(int)


        df_result_new = spark_df_result_new.select("text", "label", "prediction").toPandas()
        df_result_new['label'] = df_result_new['label'].astype(float)
        df_result_new['label'] = df_result_new['label'].astype(int)
        df_result_new['prediction'] = df_result_new['prediction'].astype(float)
        df_result_new['prediction'] = df_result_new['prediction'].astype(int)

        df_result.head(3)
        df_result_new.head(3)


        data_def = DataDefinition(
            classification=[MulticlassClassification(
                target="label",
                prediction_labels="prediction",
                # prediction_probas=[0, 1, 2],  # If probabilistic classification
                labels={0: "0.Negative", 1: "1.Positive", 2: "2.Neutral"}  # Optional, for display only
            )]
        )
        eval_data = Dataset.from_pandas(
            pd.DataFrame(df_result),
            data_definition=data_def
        )

        reference_data = Dataset.from_pandas(
            pd.DataFrame(df_result_new),
            data_definition=data_def
        )
        
        current_date = datetime.now()
        formatted_date = current_date.strftime("%d_%m_%Y")
        file_name = f"/home/ubuntu/kltn-model-monitoring/reports/Data Drift/report_{formatted_date}.html"
        logging.info(f"Saving report to {file_name}")
        
        report = Report([
            DataDriftPreset()
        ])
        
        
        datadrift_eval = report.run(eval_data, reference_data)
        datadrift_eval.save_html(file_name)
        
        ## Classification Preset
        logging.info("Running classification evaluation...")
        report = Report([
            ClassificationPreset()
            
        ],
        include_tests=True)

        classification_eval = report.run(
            current_data=eval_data, 
            reference_data=reference_data
        )
        file_name = f"/home/ubuntu/kltn-model-monitoring/reports/Model Drift/report_{formatted_date}.html"
        classification_eval.save_html(file_name)        
               
        
        
        logging.info("========== Job completed successfully ==========")
        
    except Exception as e:
        raise
    finally:
        # Clean up local log file
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    

if __name__ == "__main__":
    main()
