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


def evaluate_model(name, model, df, config, config_secret):
    """Evaluates the model on the given DataFrame and logs metrics to MLflow."""
    prediction = model.transform(df).withColumn("prediction", col("prediction")[0].cast("string"))
    prediction = prediction.withColumn("label", col("label").cast(DoubleType()))
    prediction = prediction.withColumn("prediction", col("prediction").cast(DoubleType()))
    prediction.show(3, truncate=True)
    accuracy, precision, recall, f1 = evaluator(prediction, label_col="label", prediction_col="prediction")
    
    mlflow.log_metric(f"{name}_accuracy", accuracy)
    mlflow.log_metric(f"{name}_precision", precision)
    mlflow.log_metric(f"{name}_recall", recall)
    mlflow.log_metric(f"{name}_f1", f1)
    logging.info(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    return accuracy


    
    
def tag_model_version(model_name,model_uri, model_version, test_accuracy):
    """Tags the model version in MLflow based on the test accuracy."""
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        name=model_name,
        version=str(model_version),
        key="Production",
        value="True"  

    )
    client.copy_model_version(
        src_model_uri=model_uri,
        source_version=str(model_version),
        dest_name="Production_Model",
    )
    logging.info(f"Model version {model_version} tagged as Production with accuracy: {test_accuracy}")

    
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

        mlflow.set_experiment(f"Champion Production_Model Evaluation ")
        new_model_version = get_latest_model_version(model_name="DL_Elmo_Text_Classification_Model").version
        production_model_version = get_latest_model_version(model_name="Production_Model").version
        logging.info(f"New model version: {new_model_version}, Production model version: {production_model_version}")
        

        with mlflow.start_run(run_name=f"New Model: {new_model_version} vs Production Model: {production_model_version}") as run:
            # Load model từ MLflow
            new_model, model_uri = load_model_from_mlflow(model_name, new_model_version)
            production_model, production_model_uri = load_model_from_mlflow("Production_Model", production_model_version)
            logging.info(f"Loaded new model from {model_uri} and production model from {production_model_uri}")
            # Evaluate model
            accuracy_new_model = evaluate_model(name='new_model',model=new_model, df=df_processed, config=config, config_secret=config_secret)
            accuracy_production_model = evaluate_model(name='production_model',model=production_model, df=df_processed, config=config, config_secret=config_secret)
            logging.info(f"New model accuracy: {accuracy_new_model}, Production model accuracy: {accuracy_production_model}")          
            
            
            
            # Log parameters 
            mlflow.log_param("data_test_path", data_test_path)
            mlflow.log_param("data_test_version", data_test_version)
            mlflow.log_param("data_test_count", df_processed.count())
            # Tag model version in MLflow
            if accuracy_new_model > accuracy_production_model:
                logging.info(f"New model accuracy {accuracy_new_model} is better than production model accuracy {accuracy_production_model}. Tagging new model version as Production.")
                tag_model_version(model_name, model_uri, new_model_version, accuracy_new_model)
                
            else:
                logging.info(f"New model accuracy {accuracy_new_model} is not better than production model accuracy {accuracy_production_model}. Not tagging new model version as Production.")
            

    except Exception as e:
        logging.error(f"An error occurred during the testing process: {e}")
        raise
    finally:
        logging.info("Model testing process completed.")
        
        
           

if __name__ == "__main__":
    main()

    
    # Gọi hàm và nhận DataFrame
    # df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
