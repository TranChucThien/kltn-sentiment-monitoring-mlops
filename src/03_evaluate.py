from utils.s3_process import read_csv_from_s3, push_csv_to_s3
from pyspark.ml import PipelineModel, Pipeline, Transformer
import yaml
import mlflow
import os
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "TranChucThien"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fc6e085ac3b8abf78e838bec58a25872e3db8679"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

def load_model_from_mlflow(model_name, model_version):
    # Lấy URI của mô hình đã đăng ký
    model_uri = f"models:/{model_name}/{model_version}"  # Lấy phiên bản 1 của mô hình đã register
    print(f"Loading model from {model_uri}")
    
    # Load mô hình từ MLflow
    loaded_model = mlflow.spark.load_model(model_uri)
    return loaded_model

if __name__ == "__main__":
    # Read config file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    bucket = config['s3']['bucket']
    raw_key = config['s3']['keys']['raw_data']
    input_path = f"s3a://{bucket}/{raw_key}"
    print("Input path:", input_path)
    AWS_ACCESS_KEY_ID = config['aws']['access_key']
    AWS_SECRET_ACCESS_KEY = config['aws']['secret_key']
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    
    data_validate_key = config['s3']['keys']['validate_data']
    validate_data_path = f"s3a://{bucket}/{data_validate_key}"

    # Gọi hàm và nhận DataFrame
    # df = read_csv_from_s3(validate_data_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    df = read_csv_from_s3(input_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)   
    model = load_model_from_mlflow("CountVectorizer_Model", 1)
    

    prediction = model.transform(df)
    prediction.show(3, truncate=False)