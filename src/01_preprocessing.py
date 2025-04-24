import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim, when, length
import yaml
import os
from pyspark.sql import SparkSession
import boto3
from botocore.exceptions import ClientError
import os
import shutil
from utils.clean_text import clean_text_column, preprocess

# def clean_text_column(df, input_col="Text", output_col="Text"):
#     """
#     Clean text in a DataFrame column using native Spark functions.

#     Args:
#         df (DataFrame): The input DataFrame.
#         input_col (str): The name of the column containing the raw text.
#         output_col (str): The name of the column where the cleaned text will be stored (can be the same as input_col to overwrite).

#     Returns:
#         DataFrame: DataFrame with the cleaned text column.
#     """
#     return (df.withColumn(output_col, regexp_replace(col(input_col), r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))  # Remove URLs
#               .withColumn(output_col, regexp_replace(col(output_col), r'(@|#)\w+', ''))  # Remove mentions and hashtags
#               .withColumn(output_col, lower(col(output_col)))  # Convert to lowercase
#               .withColumn(output_col, regexp_replace(col(output_col), r'[^a-zA-Z\s]', ''))  # Remove non-alphabetic characters
#               .withColumn(output_col, regexp_replace(col(output_col), r'\s+', ' '))  # Replace multiple spaces with a single space
#               )


# def preprocess(spark, config):
#     """
#     Preprocess the data including renaming columns, cleaning text, and handling labels.

#     Args:
#         spark (SparkSession): The Spark session.
#         config (dict): The configuration dictionary containing input path and other parameters.

#     Returns:
#         DataFrame: The preprocessed DataFrame.
#     """
#     # Read the CSV file into a DataFrame
#     data = spark.read.csv(config['input_path'], header=True, inferSchema=True)
#     print("Data change column name:")
#     data.show(3)

#     # Rename columns if necessary
#     data = data.withColumnRenamed("_c0", "Product")  # Rename column _c0 to Product
#     data = data.withColumnRenamed("_c1", "Label")  # Rename column _c1 to Label
#     data = data.withColumnRenamed("_c2", "Text")  # Rename column _c2 to Text
    
#     print("Data after renaming columns:")
#     data.show(3)
    
#     # Remove the header row if necessary (based on index)
#     data = data.rdd.zipWithIndex().filter(lambda row: row[1] != 0).map(lambda row: row[0]).toDF(["Product", "Label", "Text"])
    
#     # Drop rows where 'Text' column has null values
#     data = data.dropna(subset=['Text'])
    
#     # Remove rows where 'Text' or 'Label' columns are empty after trimming spaces
#     data = data.filter(trim(col("Text")) != "")
#     data = data.filter(trim(col("Label")) != "")
#     data.show(3)
    
#     # Drop the 'Product' column
#     data = data.drop("Product")
#     data.show(3)
    
#     # Clean the text in the 'Text' column
#     data = clean_text_column(data, input_col="Text", output_col="Text")
    
#     # Convert 'Label' values to numeric values (0 for Negative, 1 for Positive, 2 for Neutral, 3 for Irrelevant)
#     data = data.withColumn("Label",
#             when(col("Label") == "Negative", 0)
#           .when(col("Label") == "Positive", 1)
#           .when(col("Label") == "Neutral", 2)
#           .when(col("Label") == "Irrelevant", 3)
#           .otherwise(None)  # Fallback for unknown labels
#     )
    
#     # Remove rows where the 'Text' column is empty or 'Label' is null
#     data = data.filter(length(col("Text")) > 0)
#     data = data.withColumn("Text", trim(col("Text")))
#     data = data.filter((col("Text").isNotNull()) & (length(col("Text")) > 0))
#     data = data.filter(col("Label").isNotNull())
#     data = data.filter(col("Label").isNotNull() & (length(col("Text")) > 0))

    
#     data.show(3)
    
#     return data


if __name__ == "__main__":
    # Create SparkSession
    # spark = SparkSession.builder \
    #     .appName("Text Classification with PySpark") \
    #     .getOrCreate()

    # Read config from YAML file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("📁 Input data path:", config['data']['input_path'])

    # Preprocess the data
    data = preprocess(config['data']['input_path'] )
    
    print("📄 Cleaned data:")
    data.show(20)
# Lưu DataFrame 'data' thành một file CSV duy nhất
    print(config['data']['dataset_output_path'])
    data.coalesce(1).write.option("header", "true").mode("overwrite").csv(config['data']['dataset_output_path'])
    # Stop Spark session
    # spark.stop()

    # save_data(data, config['dataset_output_path'], format="csv")
    
    
    
    
    
    
    
        # Cấu hình thông tin AWS (đảm bảo bạn đã cấu hình chúng)
    AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
    AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
    AWS_REGION = "your-aws-region"
    BUCKET_NAME = "your-unique-bucket-name"
    S3_OUTPUT_KEY = "output_from_spark.csv"  # Tên file CSV trên S3


    AWS_ACCESS_KEY_ID = 
    AWS_SECRET_ACCESS_KEY = "
    AWS_REGION = "us-east-2"  # Ví dụ: "ap-southeast-1" (Singapore), "us-east-1" (N. Virginia)
    BUCKET_NAME = "tranchucthienops"  # Chọn một tên bucket duy nhất trên toàn cầu
    CSV_FILE_PATH = "data/twitter_training.csv"
    S3_KEY_NAME = "output.csv"  # Tên file bạn muốn lưu trên S3
    LOCAL_TEMP_DIR = "/tmp/spark_output"  
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