from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, regexp_replace, lower, trim, when, length
import nltk
from nltk.corpus import stopwords
import re
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import StringType
# Danh sách stop words ví dụ – bạn có thể dùng NLTK, spaCy, hoặc tự định nghĩa
nltk.download('stopwords')
nltk.download('punkt')

# Define English stopwords
stop_words = stopwords.words('english')
# Biểu thức regex để loại bỏ emoji
emoji_pattern = "[" \
    u"\U0001F600-\U0001F64F"  \
    u"\U0001F300-\U0001F5FF"  \
    u"\U0001F680-\U0001F6FF"  \
    u"\U0001F1E0-\U0001F1FF"  \
    u"\U00002500-\U00002BEF"  \
    u"\U00002702-\U000027B0"  \
    u"\U000024C2-\U0001F251"  \
    u"\U0001f926-\U0001f937"  \
    u"\U00010000-\U0010ffff"  \
    u"\u2640-\u2642"          \
    u"\u2600-\u2B55"          \
    u"\u200d"                 \
    u"\u23cf"                 \
    u"\u23e9"                 \
    u"\u231a"                 \
    u"\ufe0f"                 \
    u"\u3030"                 \
    "]+"

# UDF để loại bỏ stop words
@udf(StringType())
def remove_stopwords(text):
    if text:
        return ' '.join([word for word in text.split() if word not in stop_words])
    return None

def clean_text_column(df, input_col="Text", output_col="Text"):
    """
    Clean text in a Spark DataFrame column.
    """
    cleaned_df = (df.withColumn(output_col, regexp_replace(col(input_col), emoji_pattern, ''))  # Remove emojis
                    .withColumn(output_col, regexp_replace(col(output_col), r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))  # Remove URLs
                    .withColumn(output_col, regexp_replace(col(output_col), r'(@|#)\w+', ''))  # Remove mentions and hashtags
                    .withColumn(output_col, regexp_replace(col(output_col), r'<unk>', ''))  # Remove <unk>
                    .withColumn(output_col, lower(col(output_col)))  # Convert to lowercase
                    .withColumn(output_col, regexp_replace(col(output_col), r'\d+', ''))  # Remove numbers
                    .withColumn(output_col, regexp_replace(col(output_col), r'[^a-zA-Z\s]', ''))  # Remove non-alphabetic characters
                    .withColumn(output_col, regexp_replace(col(output_col), r'\s+', ' '))  # Collapse whitespace
                    # .withColumn(output_col, remove_stopwords(col(output_col)))  # Remove stop words
                 )
    return cleaned_df

# def clean_text_column(df, input_col="Text", output_col="Text"):
    """
    Clean text in a DataFrame column using native Spark functions.

    Args:
        df (DataFrame): The input DataFrame.
        input_col (str): The name of the column containing the raw text.
        output_col (str): The name of the column where the cleaned text will be stored (can be the same as input_col to overwrite).

    Returns:
        DataFrame: DataFrame with the cleaned text column.
    """
    return (df.withColumn(output_col, regexp_replace(col(input_col), r'https?://\S+|www\.\S+|\.com\S+|youtu\.be/\S+', ''))  # Remove URLs
              .withColumn(output_col, regexp_replace(col(output_col), r'(@|#)\w+', ''))  # Remove mentions and hashtags
              .withColumn(output_col, lower(col(output_col)))  # Convert to lowercase
              .withColumn(output_col, regexp_replace(col(output_col), r'[^a-zA-Z\s]', ''))  # Remove non-alphabetic characters
              .withColumn(output_col, regexp_replace(col(output_col), r'\s+', ' '))  # Replace multiple spaces with a single space
              )


def preprocess(data, input_path):
    """
    Preprocess the data including renaming columns, cleaning text, and handling labels.

    Args:
        spark (SparkSession): The Spark session.
        config (dict): The configuration dictionary containing input path and other parameters.

    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    # Read the CSV file into a DataFrame
    spark = SparkSession.builder \
        .appName("Text Classification with PySpark") \
        .getOrCreate()
    # data = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # print("Data change column name:")
    # data.show(3)

    # Rename columns if necessary
    data = data.withColumnRenamed("_c0", "Product")  # Rename column _c0 to Product
    data = data.withColumnRenamed("_c1", "Label")  # Rename column _c1 to Label
    data = data.withColumnRenamed("_c2", "Text")  # Rename column _c2 to Text
    
    # print("Data after renaming columns:")
    # data.show(3)
    
    # Remove the header row if necessary (based on index)
    data = data.rdd.zipWithIndex().filter(lambda row: row[1] != 0).map(lambda row: row[0]).toDF(["Product", "Label", "Text"])
    
    # Drop rows where 'Text' column has null values
    data = data.dropna(subset=['Text'])
    
    # Remove rows where 'Text' or 'Label' columns are empty after trimming spaces
    data = data.filter(trim(col("Text")) != "")
    data = data.filter(trim(col("Label")) != "")
    # data.show(3)
    
    # Drop the 'Product' column
    data = data.drop("Product")
    # data.show(3)
    
    # Clean the text in the 'Text' column
    data = clean_text_column(data, input_col="Text", output_col="Text")
    
    # Convert 'Label' values to numeric values (0 for Negative, 1 for Positive, 2 for Neutral, 3 for Irrelevant)
    data = data.withColumn("Label",
            when(col("Label") == "Negative", 0)
          .when(col("Label") == "Positive", 1)
          .when(col("Label") == "Neutral", 2)
          .when(col("Label") == "Irrelevant", 3)
          .otherwise(None)  # Fallback for unknown labels
    )
    
    # Remove rows where the 'Text' column is empty or 'Label' is null
    data = data.filter(length(col("Text")) > 0)
    data = data.withColumn("Text", trim(col("Text")))
    data = data.filter((col("Text").isNotNull()) & (length(col("Text")) > 0))
    data = data.filter(col("Label").isNotNull())
    data = data.filter(col("Label").isNotNull() & (length(col("Text")) > 0))

    
    # data.show(3)
    
    return data
