import boto3
from botocore.exceptions import ClientError

# Cấu hình thông tin AWS
AWS_ACCESS_KEY_ID = 
AWS_SECRET_ACCESS_KEY = 
AWS_REGION = "us-east-2"  # Ví dụ: "ap-southeast-1" (Singapore), "us-east-1" (N. Virginia)
BUCKET_NAME = "tranchucthienops"  # Chọn một tên bucket duy nhất trên toàn cầu
CSV_FILE_PATH = "data/twitter_training.csv"
S3_KEY_NAME = "output.csv"  # Tên file bạn muốn lưu trên S3

def create_s3_bucket(bucket_name, region=None):
    """Tạo một bucket S3 nếu nó chưa tồn tại."""
    try:
        if region is None:
            s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        print(f"Đã tạo bucket S3: {bucket_name} ở region {region if region else 'us-east-1'}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket '{bucket_name}' đã tồn tại và thuộc sở hữu của bạn.")
            return True
        elif e.response['Error']['Code'] == 'BucketAlreadyExists':
            print(f"Lỗi: Bucket '{bucket_name}' đã tồn tại và thuộc sở hữu của người khác.")
            return False
        else:
            print(f"Lỗi khi tạo bucket '{bucket_name}': {e}")
            return False

def upload_file_to_s3(file_path, bucket_name, key):
    """Đẩy một file lên bucket S3."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3_client.upload_file(file_path, bucket_name, key)
        print(f"Đã tải file '{file_path}' lên s3://{bucket_name}/{key}")
        return True
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'.")
        return False
    except ClientError as e:
        print(f"Lỗi khi tải file lên S3: {e}")
        return False

if __name__ == "__main__":
    # 1. Tạo bucket S3 (nếu chưa tồn tại)
    create_s3_bucket(BUCKET_NAME, AWS_REGION)

    # 2. Đẩy file CSV lên bucket S3
    upload_file_to_s3(CSV_FILE_PATH, BUCKET_NAME, S3_KEY_NAME)

    print("Hoàn tất quá trình tạo bucket (nếu cần) và tải file lên S3.")