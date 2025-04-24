from pyspark.sql import SparkSession
import boto3
from botocore.exceptions import ClientError
import os
import shutil

# Cấu hình thông tin AWS (đảm bảo bạn đã cấu hình chúng)
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
AWS_REGION = "your-aws-region"
BUCKET_NAME = "your-unique-bucket-name"
S3_OUTPUT_KEY = "output_from_spark.csv"  # Tên file CSV trên S3

AWS_ACCESS_KEY_ID = 
AWS_SECRET_ACCESS_KEY = 
AWS_REGION = "us-east-2"  # Ví dụ: "ap-southeast-1" (Singapore), "us-east-1" (N. Virginia)
BUCKET_NAME = "tranchucthienops"  # Chọn một tên bucket duy nhất trên toàn cầu
CSV_FILE_PATH = "data/twitter_training.csv"
S3_KEY_NAME = "output.csv"  # Tên file bạn muốn lưu trên S3
LOCAL_TEMP_DIR = "/tmp/spark_output"  

if __name__ == "__main__":
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
        data = spark.read.csv(CSV_FILE_PATH, header=True, inferSchema=True)
        data.printSchema()
        data.show()

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

    finally:
        # Dừng Spark Session
        spark.stop()