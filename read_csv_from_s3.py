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
from utils.s3_process import read_csv_from_s3

if __name__ == "__main__":
    input_path = 's3a://tranchucthienops/output_from_spark.csv'  # Thay đúng đường dẫn S3
    AWS_ACCESS_KEY_ID = 
    AWS_SECRET_ACCESS_KEY = 

    
    # Gọi hàm và nhận DataFrame
    df = read_csv_from_s3(input_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

    # Tiếp tục xử lý dữ liệu nếu cần
    # Ví dụ: df.show()
