from utils.s3_process import  push_csv_to_s3, read_key, upload_file_to_s3
from utils.s3_process_nlp import read_csv_from_s3, get_latest_s3_object_version
from utils.clean_text import clean_text_column
from pyspark.ml import PipelineModel, Pipeline, Transformer
from utils.mlflow_func import get_latest_model_version
import yaml
import mlflow
import os
from src._02_training import evaluator, load_config, set_up_mlflow_tracking
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def push_to_s3(df, file_name, config, config_secret):
    """Pushes a DataFrame to S3 as a CSV file."""
    bucket = config['s3']['bucket']
    data_result_key = config['s3']['keys']['test_result_new']
    result_data_path = f"s3a://{bucket}/{data_result_key}"
    
    AWS_REGION = config['aws']['region']
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    
    try:
        df.to_csv(file_name, index=False)
        upload_file_to_s3(local_path=file_name, bucket_name=bucket, s3_key=data_result_key, access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY, region=AWS_REGION)
        logging.info(f"Successfully pushed {file_name} to S3 at {result_data_path}")
    except Exception as e:
        logging.error(f"Error pushing DataFrame to S3: {e}")
        raise

def load_test_data_csv_from_s3(config, config_secret, spark):
    """Loads test data from S3."""
    bucket = config['s3']['bucket']
    data_test_key = config['s3']['keys']['test_data_new']
    test_data_path = f"s3a://{bucket}/{data_test_key}"

    AWS_REGION = config['aws']['region']
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    data_test_version = get_latest_s3_object_version(s3_path=test_data_path, aws_access_key=AWS_ACCESS_KEY_ID, aws_secret_key=AWS_SECRET_ACCESS_KEY, region=AWS_REGION)
    logging.info(f"Loading test data from {test_data_path}")
    try:
        df = read_csv_from_s3(test_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, spark)
        df.show(3, truncate=True)
        logging.info(f"Successfully loaded {df.count()} records from S3.")
        return df, test_data_path, data_test_version
    except Exception as e:
        logging.error(f"Error loading data from S3: {e}")
        raise
    

def load_model_from_mlflow(model_name, model_version):
    """Loads a registered model from MLflow."""
    model_uri = f"models:/{model_name}/{model_version}"
    logging.info(f"Loading model from model uri {model_uri}")
    try:
        loaded_model = mlflow.spark.load_model(model_uri)
        logging.info(f"Model loaded successfully from {model_uri}")
        return loaded_model, model_uri
    except Exception as e:
        logging.error(f"Error loading model from MLflow: {e}")
        raise


def evaluate_model(model, df, config, config_secret):
    """Evaluates the model on the given DataFrame and logs metrics to MLflow."""
    prediction = model.transform(df).withColumn("prediction", col("prediction")[0].cast("string"))
    prediction = prediction.withColumn("label", col("label").cast(DoubleType()))
    prediction = prediction.withColumn("prediction", col("prediction").cast(DoubleType()))
    prediction.show(3, truncate=True)
    accuracy, precision, recall, f1 = evaluator(prediction, label_col="label", prediction_col="prediction")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    df = prediction.select("text", "label", "prediction").toPandas()
    df.to_csv("predictions.csv", index=False)
    push_to_s3(df, "predictions.csv", config, config_secret)
    logging.info("Predictions saved to predictions.csv and pushed to S3.")

    
    return accuracy


    
    
def tag_model_version(model_name, model_version, test_accuracy, config):
    """Tags the model version in MLflow based on the test accuracy."""
    client = mlflow.tracking.MlflowClient()
    accuracy_threshold = config.get('evaluation', {}).get('accuracy_threshold', 0.8) # Lấy từ config, mặc định là 0.8
    if isinstance(test_accuracy, (int, float)):
        test_passed = test_accuracy > accuracy_threshold
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="New Test pass:",
            value=str(test_passed)
        )
        logging.info(f"Model version {model_version} tagged with Test pass: {test_passed} (threshold: {accuracy_threshold})")
    else:
        logging.warning(f"Test accuracy is not a number, skipping tagging for model version {model_version}")

    
def main():
    """Main function to load, test, and evaluate a registered MLflow model."""
    logging.info("Starting the model testing process...")
    model_name = "DL_Elmo_Text_Classification_Model"
    model_version = None # Set to None to use the latest version
    use_latest_version = True
    spark = SparkSession.builder \
        .appName("Load Spark NLP Model") \
        .master("local[*]") \
        .config("spark.driver.memory", "4G") \
        .config("spark.executor.memory", "4G") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3")\
        .getOrCreate()
        
    try:
        # Read config file
        config = load_config()
        config_secret = load_config(config_path="configs/secrets.yaml")
        
        # Load CSV file from S3 (raw data)
        df, data_test_path, data_test_version = load_test_data_csv_from_s3(config, config_secret, spark)
        df = df.filter(col("label") != "3")
        # Clean text column
        logging.info("Cleaning text data...")
        df_processed = clean_text_column(df, input_col="text", output_col="text")
        df_processed.show(3, truncate=True)
        logging.info("Text data cleaning complete.")

        # Set up MLflow tracking
        set_up_mlflow_tracking(config, config_secret)

        mlflow.set_experiment(f"Test {model_name}")

        if use_latest_version:
            latest_version = get_latest_model_version(model_name=model_name)
            if latest_version:
                version = latest_version.version
                logging.info(f"Using the latest model version: {version}")
            else:
                logging.warning(f"No version found for model '{model_name}'.")
                return # Exit if no model version found
        elif specific_model_version is not None:
            version = specific_model_version
            logging.info(f"Using specified model version: {version}")
        else:
            logging.info(f"Using default model version: 1")
            version = 1

        with mlflow.start_run(run_name=f"Test {model_name} version {version}"):
            # Load model từ MLflow
            model, model_uri = load_model_from_mlflow(model_name, version)

            # Evaluate model
            accuracy = evaluate_model(model, df_processed, config=config, config_secret=config_secret)
            
            logging.info(f"Model evaluation completed with accuracy: {accuracy}")
            
            # Log parameters 
            mlflow.log_param("data_test_path", data_test_path)
            mlflow.log_param("data_test_version", data_test_version)
            mlflow.log_param("data_test_count", df_processed.count())
            # Tag model version in MLflow
            tag_model_version(model_name, version, accuracy, config)

    except Exception as e:
        logging.error(f"An error occurred during the testing process: {e}")
        raise
    finally:
        logging.info("Model testing process completed.")
        
        
           

if __name__ == "__main__":
    main()

    
    # Gọi hàm và nhận DataFrame
    # df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
