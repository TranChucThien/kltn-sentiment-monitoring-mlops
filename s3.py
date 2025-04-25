from pyspark.sql import SparkSession
import boto3
from botocore.exceptions import NoCredentialsError

# Cấu hình thông tin S3
AWS_ACCESS_KEY_ID = "AKIAXFPLNKJI2R25A5VM"
AWS_SECRET_ACCESS_KEY = "KFEHQXgqg3fsR0QOXASi76QBpBK6kZUNSvULSmc4"
S3_SOURCE_BUCKET = "your-source-bucket-name"
S3_SOURCE_KEY = "path/to/your/input_file.txt"
S3_DESTINATION_BUCKET = "your-destination-bucket-name"
S3_DESTINATION_KEY = "path/to/your/output_file.txt"

# Khởi tạo Spark Session
spark = SparkSession.builder.appName("S3FileTransfer").getOrCreate()

# Cấu hình AWS credentials cho Spark (nếu cần xử lý dữ liệu bằng Spark)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", AWS_ACCESS_KEY_ID)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true") # Cần thiết cho một số region và cấu hình S3
# spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.your-region.amazonaws.com") # Nếu bạn đang sử dụng region không phải us-east-1

def download_from_s3(bucket_name, key, local_path):
    """Tải file từ S3 về local."""
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3.download_file(bucket_name, key, local_path)
        print(f"Đã tải file từ s3://{bucket_name}/{key} về {local_path}")
        return True
    except NoCredentialsError:
        print("Lỗi: Chưa tìm thấy thông tin xác thực AWS.")
        return False
    except Exception as e:
        print(f"Lỗi khi tải file từ S3: {e}")
        return False

def upload_to_s3(local_path, bucket_name, key):
    """Đẩy file từ local lên S3."""
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3.upload_file(local_path, bucket_name, key)
        print(f"Đã tải file từ {local_path} lên s3://{bucket_name}/{key}")
        return True
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {local_path}")
        return False
    except NoCredentialsError:
        print("Lỗi: Chưa tìm thấy thông tin xác thực AWS.")
        return False
    except Exception as e:
        print(f"Lỗi khi tải file lên S3: {e}")
        return False

if __name__ == "__main__":
    local_file_path = "temp_downloaded_file.txt"

    # Tải file từ S3
    if download_from_s3(S3_SOURCE_BUCKET, S3_SOURCE_KEY, local_file_path):
        # --- Xử lý dữ liệu bằng Spark (ví dụ) ---
        df = spark.sparkContext.textFile(f"file:///{local_file_path}")
        # Thực hiện các phép biến đổi dữ liệu nếu cần
        # Ví dụ: Đếm số dòng
        line_count = df.count()
        print(f"Số dòng trong file: {line_count}")

        # Lưu kết quả (hoặc file đã xử lý) xuống local nếu cần
        output_local_path = "temp_processed_file.txt"
        df.saveAsTextFile(f"file:///{output_local_path}")

        # Đẩy file (đã xử lý hoặc file gốc) lên S3
        upload_to_s3(output_local_path, S3_DESTINATION_BUCKET, S3_DESTINATION_KEY)

        # Hoặc đẩy file gốc đã tải về lên S3
        # upload_to_s3(local_file_path, S3_DESTINATION_BUCKET, S3_DESTINATION_KEY)

        # Xóa file tạm
        import os
        os.remove(local_file_path)
        os.remove(output_local_path) # Nếu có file đã xử lý

    # Dừng Spark Session
    spark.stop()