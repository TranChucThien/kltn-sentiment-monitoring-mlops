from utils.s3_process import read_csv_from_s3, push_csv_to_s3, read_key
from utils.clean_text import clean_text_column
from pyspark.ml import PipelineModel, Pipeline, Transformer
from utils.mlflow_func import get_latest_model_version
import yaml
import mlflow
import os
from src._02_training import evaluator
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="configs/config.yaml"):
    """Loads configuration from the specified YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuration loaded successfully from {config_path}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

def load_test_data_csv_from_s3(config, config_secret):
    """Loads test data from S3."""
    bucket = config['s3']['bucket']
    data_test_key = config['s3']['keys']['test_data']
    test_data_path = f"s3a://{bucket}/{data_test_key}"
    AWS_REGION = config['aws']['region']
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    logging.info(f"Loading test data from {test_data_path}")
    try:
        df = read_csv_from_s3(test_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        df.show(3, truncate=True)
        logging.info(f"Successfully loaded {df.count()} records from S3.")
        return df
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


def evaluate_model(model, df):
    """Evaluates the model on the given DataFrame and logs metrics to MLflow."""
    prediction = model.transform(df)
    prediction.show(3, truncate=True)
    accuracy, precision, recall, f1 = evaluator(prediction)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    return accuracy

def set_up_mlflow_tracking(config, config_secret):
    """Sets up MLflow tracking URI and credentials."""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config_secret['mlflow']['password']
    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow']['tracking_uri']
    os.environ['MLFLOW_TRACKING_USERNAME'] = config['mlflow']['username']
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    logging.info("MLflow tracking setup complete.")
    
    
def tag_model_version(model_name, model_version, test_accuracy):
    """Tags the model version in MLflow based on the test accuracy."""
    client = mlflow.tracking.MlflowClient()
    accuracy_threshold = config.get('evaluation', {}).get('accuracy_threshold', 0.8) # Lấy từ config, mặc định là 0.8
    if isinstance(test_accuracy, (int, float)):
        test_passed = test_accuracy > accuracy_threshold
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="Test pass:",
            value=str(test_passed)
        )
        logging.info(f"Model version {model_version} tagged with Test pass: {test_passed} (threshold: {accuracy_threshold})")
    else:
        logging.warning(f"Test accuracy is not a number, skipping tagging for model version {model_version}")

    
def main(model_name, model_version=1, use_latest_version=True):
    """Main function to load, test, and evaluate a registered MLflow model."""
    logging.info("Starting the model testing process...")
    try:
        # Read config file
        config = load_config()
        config_secret = load_config(config_path="configs/secrets.yaml")
        
        # Load CSV file from S3 (raw data)
        df = load_test_data_csv_from_s3(config, config_secret)

        # Clean text column
        logging.info("Cleaning text data...")
        df_processed = clean_text_column(df)
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
            accuracy = evaluate_model(model, df_processed)

            # Tag model version in MLflow
            tag_model_version(model_name, version, accuracy, config)

    except Exception as e:
        logging.error(f"An error occurred during the testing process: {e}")
    finally:
        logging.info("Model testing process finished.")
        
        
           

if __name__ == "__main__":
    main(model_name="CountVectorizer_Model", use_latest_version=True)
    main(model_name="HashingTF_IDF_Model", use_latest_version=True)
    
    # Gọi hàm và nhận DataFrame
    # df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
