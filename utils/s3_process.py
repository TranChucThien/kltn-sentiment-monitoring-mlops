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


def read_key(path):
    df = pd.read_csv(path)
    AWS_ACCESS_KEY_ID = df['Access key ID'][0]
    AWS_SECRET_ACCESS_KEY = df['Secret access key'][0]
    return AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def read_csv_from_s3(input_path: str, aws_access_key: str, aws_secret_key: str, region: str ):
    """
    Hàm đọc dữ liệu CSV từ S3 sử dụng PySpark.

    :param input_path: Đường dẫn đến file CSV trên S3 (ví dụ: s3a://bucket-name/path/to/file.csv)
    :param aws_access_key: AWS Access Key
    :param aws_secret_key: AWS Secret Key
    :param region: Vùng S3, mặc định là 'us-east-2'
    :return: DataFrame chứa dữ liệu CSV từ S3
    """
    
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("Read CSV from S3 with PySpark") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .getOrCreate()
    
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
