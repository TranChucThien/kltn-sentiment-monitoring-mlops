# import findspark
# findspark.init()  
# from pyspark.sql import SparkSession

# # Config chứa đường dẫn file S3
# config = {
#     'input_path': 's3a://tranchucthienops/output_from_spark.csv'  # Thay đúng bucket và key
# }

# # Khởi tạo SparkSession
# spark = SparkSession.builder \
#     .appName("Read CSV from S3 with PySpark") \
#     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
#     .getOrCreate()
# AWS_ACCESS_KEY_ID = "aaa"
# AWS_SECRET_ACCESS_KEY = "aaaa"
# # Cấu hình S3 (vùng us-east-2 - Ohio)
# hadoopConf = spark._jsc.hadoopConfiguration()
# hadoopConf.set("fs.s3a.access.key", AWS_ACCESS_KEY_ID)
# hadoopConf.set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
# hadoopConf.set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
# hadoopConf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# # Đọc dữ liệu CSV từ S3
# data = spark.read.csv(config['input_path'], header=True, inferSchema=True)

# # Hiển thị dữ liệu
# data.show()

# # Dừng Spark
# spark.stop()



import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
import yaml
import boto3
import shutil
from botocore.exceptions import ClientError
from pyspark.sql.functions import col, trim, length, when
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import boto3
import pandas as pd
import boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError # Important: Import ClientError from botocore.exceptions

def read_key(path):
    df = pd.read_csv(path)
    AWS_ACCESS_KEY_ID = df['Access key ID'][0]
    AWS_SECRET_ACCESS_KEY = df['Secret access key'][0]
    return AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def read_csv_from_s3(input_path: str, aws_access_key: str, aws_secret_key: str, region: str , spark):
    """
    Hàm đọc dữ liệu CSV từ S3 sử dụng PySpark.

    :param input_path: Đường dẫn đến file CSV trên S3 (ví dụ: s3a://bucket-name/path/to/file.csv)
    :param aws_access_key: AWS Access Key
    :param aws_secret_key: AWS Secret Key
    :param region: Vùng S3, mặc định là 'us-east-2'
    :return: DataFrame chứa dữ liệu CSV từ S3
    """
    
    # Khởi tạo SparkSession
    # spark = SparkSession.builder \
    #     .appName("Read CSV from S3 with PySpark") \
    #     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    #     .getOrCreate()
    
    # Cấu hình S3 với AWS Access Key và Secret Key
    hadoopConf = spark._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.access.key", aws_access_key)
    hadoopConf.set("fs.s3a.secret.key", aws_secret_key)
    hadoopConf.set("fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
    hadoopConf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Đọc dữ liệu CSV từ S3
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Hiển thị dữ liệu
    # data.show()

    # # Dừng Spark
    # spark.stop()

    return data


def push_csv_to_s3(data, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME, S3_OUTPUT_KEY, LOCAL_TEMP_DIR="/tmp/spark_output"):
         # Khởi tạo Spark Session
    spark = SparkSession.builder.appName("SparkUploadToS3") \
        .config("spark.local.dir", LOCAL_TEMP_DIR) \
        .getOrCreate()

    # Cấu hình AWS credentials cho Spark
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", AWS_ACCESS_KEY_ID)
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
    
    if AWS_REGION not in ['us-east-1', None]:
        spark._jsc.hadoopConfiguration().set(f"fs.s3a.endpoint.{AWS_REGION}.amazonaws.com", f"s3.{AWS_REGION}.amazonaws.com")

    # Tạo thư mục tạm nếu nó chưa tồn tại
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    try:
        # Giả sử bạn đã có DataFrame 'data' từ việc đọc file CSV
        # data = spark.read.csv(CSV_FILE_PATH, header=True, inferSchema=True)
        data.printSchema()
        data.show(5)

        # 1. Lưu DataFrame xuống file CSV trong WSL filesystem (tạm thời)
        local_output_path = os.path.join(LOCAL_TEMP_DIR, "processed_data")
        output_uri = f"file:///{local_output_path}"

        data.write.csv(output_uri, header=True, mode="overwrite")

        # Spark lưu thành một thư mục chứa nhiều part file, chúng ta cần tìm file .csv thực tế
        output_dir = local_output_path
        part_files = [f for f in os.listdir(output_dir) if f.startswith("part-") and f.endswith(".csv")]

        if part_files:
            # Lấy đường dẫn của file part đầu tiên
            actual_output_file_path = os.path.join(output_dir, part_files[0])

            # 2. Sử dụng boto3 để upload file CSV từ local lên S3
            s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
            try:
                s3_client.upload_file(actual_output_file_path, BUCKET_NAME, S3_OUTPUT_KEY)
                print(f"Đã tải DataFrame lên s3://{BUCKET_NAME}/{S3_OUTPUT_KEY}")
            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy file '{actual_output_file_path}'.")
            except ClientError as e:
                print(f"Lỗi khi tải file lên S3: {e}")

            # 3. Dọn dẹp file và thư mục tạm
            os.remove(actual_output_file_path)
            shutil.rmtree(local_output_path, ignore_errors=True)
        else:
            print("Lỗi: Không tìm thấy file CSV part nào sau khi Spark ghi.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

    # finally:
    #     # # Dừng Spark Session
    #     # spark.stop()


def upload_file_to_s3(local_path, bucket_name, s3_key, access_key, secret_key, region):
    s3_client = boto3.client('s3',
                             aws_access_key_id=access_key,
                             aws_secret_access_key=secret_key,
                             region_name=region)
    s3_client.upload_file(local_path, bucket_name, s3_key)    

def get_latest_s3_object_version(s3_path: str, aws_access_key: str, aws_secret_key: str, region: str) -> dict:
    """
    Retrieves information about the LAST (latest) version of an S3 object.

    Args:
        s3_path (str): The S3 path of the object (e.g., "s3a://your-bucket-name/path/to/file.csv").
        aws_access_key (str): Your AWS Access Key ID.
        aws_secret_key (str): Your AWS Secret Access Key.
        region (str): The AWS Region of the bucket (e.g., "ap-southeast-1").

    Returns:
        dict: A dictionary containing information about the latest version of the object.
              If an error occurs or the latest version is not found, the function will
              print an error message and return an empty dictionary.
    """
    try:
        # 1. Parse the S3 path to extract the bucket name and object key
        parsed_url = urlparse(s3_path)
        if parsed_url.scheme not in ['s3', 's3a', 's3n']:
            print(f"Error: Scheme '{parsed_url.scheme}' is not supported. Please use 's3://', 's3a://', or 's3n://'.")
            return {}

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip('/') # Remove the leading '/' from the path

        if not bucket_name or not object_key:
            print(f"Error: Could not parse bucket name or object key from path '{s3_path}'.")
            return {}

        # 2. Initialize the Boto3 S3 Client with provided credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )

        # 3. Use list_object_versions() to retrieve object versions
        # The 'Prefix' parameter is used to filter by the object's key.
        response = s3_client.list_object_versions(
            Bucket=bucket_name,
            Prefix=object_key
        )

        latest_version = {}

        # Iterate through the versions and find the one with 'IsLatest' set to True
        if 'Versions' in response:
            for version in response['Versions']:
                # Ensure the key matches exactly and check if it's the latest
                if version['Key'] == object_key and version.get('IsLatest'):
                    latest_version = version
                    break # Found the latest version, so stop iterating

        # If no latest version was found in 'Versions' but there are 'DeleteMarkers',
        # check if the latest delete marker is marked as 'IsLatest'.
        # This occurs if the object was recently deleted, and the delete marker is the newest change.
        if not latest_version and 'DeleteMarkers' in response:
            for delete_marker in response['DeleteMarkers']:
                if delete_marker['Key'] == object_key and delete_marker.get('IsLatest'):
                    latest_version = delete_marker
                    break
        
        return latest_version.get('VersionId', 'N/A')

    except ClientError as e: # Catch ClientError from botocore.exceptions
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"AWS Client Error: [{error_code}] {error_message}")
        print(f"Please check your AWS Access Key, Secret Key, Region, and permissions (s3:ListBucketVersions is required).")
        return {}
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return {}
    
# Ví dụ về input và output:
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
    
    # Gọi hàm và nhận DataFrame
    df = read_csv_from_s3(input_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)

    push_csv_to_s3(df, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME, S3_OUTPUT_KEY, LOCAL_TEMP_DIR='/tmp/spark_output')
