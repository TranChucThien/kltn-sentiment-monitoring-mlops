from utils.s3_process import read_csv_from_s3, push_csv_to_s3, read_key
from utils.clean_text import clean_text_column
from pyspark.ml import PipelineModel, Pipeline, Transformer
import yaml
import mlflow
import os
from src._02_training import evaluator

def load_model_from_mlflow(model_name, model_version):
    # Lấy URI của mô hình đã đăng ký
    model_uri = f"models:/{model_name}/{model_version}"  # Lấy phiên bản 1 của mô hình đã register
    print(f"Loading model from {model_uri}")
    
    # Load mô hình từ MLflow
    loaded_model = mlflow.spark.load_model(model_uri)
    return loaded_model, model_uri

def main(name, version=1):
    # Read config file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    bucket = config['s3']['bucket']
    raw_key = config['s3']['keys']['raw_data']
    input_path = f"s3a://{bucket}/{raw_key}"
    print("Input path:", input_path)

    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    os.environ["MLFLOW_TRACKING_PASSWORD"]= config['mlflow']['password'] 
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config['aws']['access_key_path'])
    data_validate_key = config['s3']['keys']['validate_data']
    validate_data_path = f"s3a://{bucket}/{data_validate_key}"
    os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "TranChucThien"
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)   
    df.show(3, truncate=True)
    df = clean_text_column(df) 
    df.show(3, truncate=True)
    mlflow.set_experiment(f"Evaluation {name}" )
    with mlflow.start_run(run_name=f"Evaluation {name} version {version}"):
        # Load model từ MLflow
        model, model_uri = load_model_from_mlflow(name, version)
        prediction = model.transform(df)
        prediction.show(3, truncate=True)
        accuracy, precision, recall, f1 = evaluator(prediction)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        print(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
        
        # Tag model version in MLflow
        model_name = name
        model_version = version
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="Validation",
            value="True" if accuracy > 0.8 else "False"
            )
        
        
        

if __name__ == "__main__":
    main(name="CountVectorizer_Model", version=1)
    main(name="HashingTF_IDF_Model", version=1)

    # Gọi hàm và nhận DataFrame
    # df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
