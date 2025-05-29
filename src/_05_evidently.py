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
import json

from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def send_drift_notification_email(sender_email, sender_password, receiver_email, fail_info):
    """
    Sends an email notification for model drift (English, concise).

    Args:
        sender_email (str): The sender's email address.
        sender_password (str): The sender's app password.
        receiver_email (str): The recipient's email address for the alert.
        fail_info (str): A string containing detailed drift failure information.
    """
    try:
        # Email Subject
        subject = "🚨 ALERT: Model Drift Detected!"

        # Email Body (concise HTML)
        body = f"""
        <html>
        <body>
            <p>Dear Team,</p>
            <p>This is an automated alert. We've detected **potential model drift** based on recent performance metrics.</p>
            <p>Please review the detailed information below:</p>
            <p><strong>Drift Failure Information:</strong></p>
            <p style="font-family: monospace; color: red; font-weight: bold;">
            The following metrics have failed the drift test:</p>
            <
            <pre style="background-color: #f2f2f2; padding: 10px; border-radius: 5px; font-family: monospace;">{fail_info}</pre>
            
            <p>Further investigation into the model's behavior and input data is recommended.</p>
            <p>Regards,</p>
            <p>Automated Model Monitoring System</p>
        </body>
        </html>
        """

        msg = MIMEText(body, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print("Model drift notification email sent successfully!")

    except smtplib.SMTPAuthenticationError:
        print("Authentication error: Incorrect username/password, or you need to enable 'App Passwords' for your Google account.")
        print("Please check your Google account security settings (or your email provider's settings).")
    except smtplib.SMTPConnectError as e:
        print(f"SMTP connection error: {e}")
        print("This might be due to an unavailable SMTP server or network issues.")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
    finally:
        if 'server' in locals() and server:
            server.quit()


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
        
        email_password = config_secret['email']['password']
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
            pd.DataFrame(df_result_new),
            data_definition=data_def
        )

        reference_data = Dataset.from_pandas(
            pd.DataFrame(df_result),
            data_definition=data_def
        )
        
        current_date = datetime.now()
        formatted_date = current_date.strftime("%d_%m_%Y_%H_%M_%S")
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
               
        report_json_str = classification_eval.json()
        report_json = json.loads(report_json_str)
        fail_infor =""
        num_fail = 0
        for test in report_json["tests"]:
            if test["status"] == "FAIL":
                num_fail += 1
                fail_infor += f"{num_fail}. ⚠️ {test['name']}  FAILED\n"
                fail_infor += f"   📌 Description: {test['description']}\n"
                
        print(f"Total number of failed tests: {num_fail}")
        print(fail_infor)
        my_email = "tranchucthienmt@gmail.com"  # Your sender email address
        # my_password = os.getenv("EMAIL_PASSWORD") # Your app password (for Gmail)
        my_password = email_password # Your app password (for Gmail)
        recipient_email = "tranchucthienmt@gmail.com" # The recipient's email address
            
        if num_fail > 0:
            logging.info("Drift detected, sending notification email...")
            # Send email notification
            send_drift_notification_email(my_email, my_password, recipient_email, fail_infor)
        else:
            logging.info("No drift detected, no email sent.")
        
        logging.info("========== Job completed successfully ==========")
        
    except Exception as e:
        raise
    finally:
        # Clean up local log file
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    

if __name__ == "__main__":
    main()
