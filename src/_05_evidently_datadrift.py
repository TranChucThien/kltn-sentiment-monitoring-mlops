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
from evidently.tests import *
from datetime import datetime
import json

from datetime import datetime
import smtplib
from email.mime.text import MIMEText

import subprocess
import json
import argparse
import os

from pymongo import MongoClient


from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import json
import ast


def get_max_confidence_from_string_representation(category_str):
    if category_str:
        try:

            
            start_index = category_str.find("metadata=")
            if start_index != -1:
                metadata_str_start = start_index + len("metadata=")
                # Finding the end of the metadata dictionary
                brace_count = 0
                end_index = -1
                for i in range(metadata_str_start, len(category_str)):
                    if category_str[i] == '{':
                        brace_count += 1
                    elif category_str[i] == '}':
                        brace_count -= 1
                    
                    if brace_count == 0 and category_str[i] == '}':
                        end_index = i + 1
                        break
                
                if end_index != -1:
                    metadata_dict_str = category_str[metadata_str_start:end_index]
                    
                    # Use ast.literal_eval to safely parse the dictionary string
                    metadata = ast.literal_eval(metadata_dict_str)
                    
                    confidences = []
                    for key, value_str in metadata.items():
                        # Check if the key is a digit
                        if key.isdigit(): 
                            try:
                                confidences.append(float(value_str))
                            except ValueError:
                                continue 
                    
                    if confidences:
                        return max(confidences)
        except (SyntaxError, ValueError, IndexError, AttributeError):
            # If there's an error in parsing, return 0.0
            return 0.0
    return 0.0


get_max_confidence_udf_single_cell = udf(get_max_confidence_from_string_representation, DoubleType())

def save_to_mongo(report_json, db_name: str, collection_name: str):
    """
    Saves the report JSON to MongoDB.

    Args:
        report (str): The report JSON string.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
    """
    

    client = MongoClient("mongodb+srv://admin:01122003@cluster0.atbocxy.mongodb.net/")
    db = client[db_name]
    collection = db[collection_name]
    try:
        # document = {
        #     "timestamp": datetime.now().isoformat(),
        #     "report": report_json
        # }
        report_json = json.loads(report_json)  # Ensure report_json is a dictionary
        report_json["timestamp"] = datetime.now().isoformat()
        
        # Insert the report into the collection
        collection.insert_one(report_json)
        print(f"✅ Report successfully saved to MongoDB in {db_name}.{collection_name}")
        logging.info(f"Report successfully saved to MongoDB in {db_name}.{collection_name}")
    except Exception as e:
        print(f"❌ Error saving report to MongoDB: {e}")
        logging.error(f"Error saving report to MongoDB: {e}")
        
        
def append_alert_to_log(name: str, description: str, recipient_email: str, file_path: str = "alerts.log"):
    """
    Logs a JSON alert to a file.

    Args:
        name (str): The alert's name or summary content.
        description (str): Detailed description of the alert.
        recipient_email (str): The email address of the alert recipient.
        file_path (str): The path to the log file (defaults to alerts.log).
    """
    alert_entry = {
        "timestamp": datetime.now().isoformat(),
        "alert_type": "Data Drift Detected",
        "status": "Sent",
        "details": {
            "recipient": recipient_email,
            "name": name,
            "description": description
        }
    }

    try:
        with open(file_path, "a", encoding="utf-8") as file:
            json.dump(alert_entry, file, ensure_ascii=False)
            file.write("\n")
        print(f"✅ Alert successfully logged to {file_path}")
    except Exception as e:
        print(f"❌ Error logging alert: {e}")

def curl(run_option:str, clean_infra='false', provision_infra='true', token=None):
    """
    Function to send a POST request to the GitHub Actions API to trigger a workflow.
    
    :param run_option: Run option for the pipeline
    :param clean_infra: Whether to clean the infrastructure or not
    :param provision_infra: Whether to provision the infrastructure or not
    """
    # Lấy token từ biến môi trường
    # token = os.getenv("GITHUB_TOEKN")
    token = token
    if not token:
        raise ValueError("You need to set the GITHUB_TOKEN environment variable")

    owner = "TranChucThien"
    repo = "kltn-sentiment-monitoring-mlops"
    workflow_file_name = "MLOPS_Pipeline_v3.yml"
    ref = "main"

    # URL API
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file_name}/dispatches"

    # Dữ liệu gửi đi
    data = {
        "ref": ref,
        "inputs": {
            "run_option": run_option,
            "clean_infra": clean_infra,
            "provision_infra": provision_infra,
        }
    }

    # Gọi API qua curl
    command = [
        "curl",
        "-X", "POST",
        "-H", f"Authorization: token {token}",
        "-H", "Accept: application/vnd.github.v3+json",
        "-d", json.dumps(data),
        url
    ]

    # Chạy lệnh
    subprocess.run(command)

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
        subject = "🚨 ALERT: Data Drift Detected!"

        # Email Body (concise HTML)
        body = f"""
        <html>
        <body>
            <p>Dear Team,</p>
            <p>This is an automated alert. We've detected **potential data drift** based on recent performance metrics.</p>
            <p>Please review the detailed information below:</p>
            <p><strong>Drift Failure Information:</strong></p>
            <p style="font-family: monospace; color: red; font-weight: bold;">
            The following metrics have failed the drift test:</p>
            
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
        print("Data drift notification email sent successfully!")

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
        github_token = config_secret['github']['token']
        bucket = config['s3']['bucket']
        test_result_key = config['s3']['keys']['datadrift_reference']
        test_result_new_key = config['s3']['keys']['datadrift_eval']

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
   
   
        spark_df_result_new = spark_df_result_new.withColumn(
            "confidence", get_max_confidence_udf_single_cell(spark_df_result_new["category"])
        )

        spark_df_result_new.select("text", "prediction", "confidence").show(2,truncate=False)

        spark_df_result = spark_df_result.withColumn(
            "confidence", get_max_confidence_udf_single_cell(spark_df_result["category"])
        )
        spark_df_result.select("text", "prediction", "confidence").show(2, truncate=False)

        # Convert columns to StringType
        logging.info("Converting columns to StringType...")
        spark_df_result_ = spark_df_result.withColumn("label", col("label").cast(StringType()))
        spark_df_result_ = spark_df_result.withColumn("prediction", col("prediction").cast(StringType()))

        spark_df_result_new_ = spark_df_result_new.withColumn("label", col("label").cast(StringType()))
        spark_df_result_new_ = spark_df_result_new.withColumn("prediction", col("prediction").cast(StringType()))
        logging.info("Columns converted to StringType successfully")

        # Preprocess text data
        logging.info("Preprocessing text data...")
        df_result = spark_df_result_.select("text", "prediction", "sentence_embeddings", "confidence").toPandas()
        # df_result = spark_df_result_.select("prediction",  "confidence").toPandas()
        # df_result['label'] = df_result['label'].astype(float)
        # df_result['label'] = df_result['label'].astype(int)
        df_result['prediction'] = df_result['prediction'].astype(float)
        df_result['prediction'] = df_result['prediction'].astype(int)


        #df_result_new = spark_df_result_new_.select("prediction",  "confidence").toPandas()
        df_result_new = spark_df_result_new_.select("text", "prediction", "sentence_embeddings", "confidence").toPandas()
        # df_result_new['label'] = df_result_new['label'].astype(float)
        # df_result_new['label'] = df_result_new['label'].astype(int)
        df_result_new['prediction'] = df_result_new['prediction'].astype(float)
        df_result_new['prediction'] = df_result_new['prediction'].astype(int)

        df_result.head(3)
        df_result_new.head(3)
        logging.info("Text data preprocessed successfully")

        logging.info("Starting Evidently Data Drift evaluation...")
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
        file_name = f"report_{formatted_date}.html"
        file_name = f"/home/ubuntu/kltn-model-monitoring/reports/Data Drift/report_{formatted_date}.html"
        logging.info(f"Saving report to {file_name}")
        
        report = Report([
            DataDriftPreset(
                # drift_share= 0.76,
                # method="psi"
                cat_method="psi",
                cat_threshold= 0.1,
                num_method= "psi",
                num_threshold=0.1,
                text_method="abs_text_content_drift",
                text_threshold=0.5,
                
            )

            
        ], include_tests=True)
        
        
        datadrift_eval = report.run(eval_data, reference_data)
        datadrift_eval.save_html(file_name)
        
        report_json_str = datadrift_eval.json()
        save_to_mongo(report_json=report_json_str, db_name="reports", collection_name="data_drift")
        report_json = json.loads(report_json_str)
        fail_infor =""
        num_fail = 0
        for test in report_json["tests"]:
            if test["status"] == "FAIL":
                num_fail += 1
                fail_infor += f"{num_fail}. {test['name']}  FAILED\n"
                fail_infor += f"Description: {test['description']}\n"
                append_alert_to_log(
                    name=test['name'],
                    description=test['description'],
                    recipient_email="tranchucthienmt@gmail.com",
                    file_path="/home/ubuntu/kltn-model-monitoring/alert/alerts.log"
                )
                document = {
                    "type": "Data Drift Detected",
                    "test_case": test['name'],
                    "description": test['description'],
                    "status": test['status'],

                }
                save_to_mongo(report_json=json.dumps(document), db_name="reports", collection_name="alerts")
                
        
        logging.info(f"Total number of failed tests: {num_fail}")      
        print(f"Total number of failed tests: {num_fail}")
        print(fail_infor)
        logging.info(f"Fail information: {fail_infor}")
        # Send email notification if drift is detected
        my_email = "tranchucthienmt@gmail.com"  # Your sender email address
        # my_password = os.getenv("EMAIL_PASSWORD") # Your app password (for Gmail)
        my_password = email_password # Your app password (for Gmail)
        recipient_email = "tranchucthienmt@gmail.com" # The recipient's email address
            
        if num_fail > 0:
            logging.info("Drift detected, sending notification email...")
            # Send email notification
            send_drift_notification_email(my_email, my_password, recipient_email, fail_infor)
            logging.info("Email sent successfully!")
            
            # logging.info("Drift detected, retrigger pipeline...")
            # curl("train", clean_infra='false', provision_infra='true', token=github_token)
            
            # document = {
            #     "type": "Model Drift Detected", 
            #     "num_fail": num_fail,
            #     "fail_info": fail_infor,
                
            # }
            # # Save 
            # save_to_mongo(report_json=json.dumps(document), db_name="reports", collection_name="alerts")
        else:
            logging.info("No drift detected, no email sent, no trigger.")
        
        logging.info("========== Job completed successfully ==========")
        
    except Exception as e:
        raise
    finally:
        # Clean up local log file
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    

if __name__ == "__main__":
    main()
