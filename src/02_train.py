import findspark
findspark.init()
from utils.s3_process import read_csv_from_s3
import yaml

# Thêm các thư viện cần thiết
import nltk
from nltk.corpus import stopwords

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
# Các import khác...

from pyspark.sql import SparkSession


if __name__ == "__main__":
    # Load configuration
    #  Read config file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    print("Input path:", dataset_path)
    # AWS credentials and region
    AWS_ACCESS_KEY_ID = config['aws']['access_key']
    AWS_SECRET_ACCESS_KEY = config['aws']['secret_key']
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    
    data = read_csv_from_s3(dataset_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    print("Read csv file from S3, this is raw data")
    data.show(3)
    
    
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Test data:")
    test_data.printSchema()
    test_data.show(3)
    
    train_data = data
    
    
    nltk.download('stopwords')
    nltk.download('punkt')

    # Define English stopwords
    stop_words = stopwords.words('english')

    print("Stopwords:", stop_words)
    # Bước 1: Tách từ
    tokenizer = Tokenizer(inputCol="Text", outputCol="words")

    # Bước 2: Loại bỏ stopwords (sử dụng danh sách NLTK)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stop_words)

    # Bước 3: Vector hóa văn bản
    vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

    # Bước 4: Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="Label", maxIter=20)

    # Bước 5: Pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, lr])
    # Huấn luyện mô hình
    model1 = pipeline.fit(train_data)
    
    prediction1 = model1.transform(test_data)
    # # Hiển thị kết quả mẫu
    prediction1.select("Text", "Label", "prediction", "probability").show(20, truncate=True)
    

    evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction")
    accuracy = evaluator.evaluate(prediction1, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(prediction1, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(prediction1, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(prediction1, {evaluator.metricName: "f1"})

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    