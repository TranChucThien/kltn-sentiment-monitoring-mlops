
from utils.s3_process import read_csv_from_s3, push_csv_to_s3
from utils.clean_text import preprocess
import yaml

if __name__ == "__main__":
    #  Read config file
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
    
    # Read csv file from S3, this is raw data
    df = read_csv_from_s3(input_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    print("Read csv file from S3, this is raw data")
    df.show(3)
    
    # Preprocess data
    print("Preprocess data")
    df = preprocess(df, input_path)
    
    # Show the preprocessed data
    print("Preprocessed data:")
    df.printSchema()
    df.show(3)
    
    # Push the preprocessed data to S3
    print("Push the preprocessed data to S3")
    push_csv_to_s3(df, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME, S3_OUTPUT_KEY, LOCAL_TEMP_DIR='/tmp/spark_output')
