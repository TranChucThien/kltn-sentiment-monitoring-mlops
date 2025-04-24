from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, regexp_replace, lower, trim, when, length


def clean_text_column(df, input_col="Text", output_col="Text"):
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
