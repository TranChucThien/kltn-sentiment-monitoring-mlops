import findspark
findspark.init()
from utils.s3_process import read_csv_from_s3, read_key
from utils.clean_text import clean_text_column

from utils.mlflow_func import get_latest_model_version, get_model_version_by_stage
import datetime
import yaml
import mlflow
import os 


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="TranChucThien"


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
from pyspark.sql import Row
from pyspark.sql.functions import col, when, lit, udf, regexp_replace, lower, trim
# 
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

# 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
from multiprocessing import Process
import logging

# def add_class_weights(data, label_col="Label"):
#     # Đếm số lượng mẫu trong từng lớp
#     label_counts = data.groupBy(label_col).count().toPandas()
#     total = label_counts["count"].sum()

#     # Tính trọng số ngược tỷ lệ với tần suất xuất hiện
#     weights = {row[label_col]: total / row["count"] for _, row in label_counts.iterrows()}

#     # Thêm cột trọng số vào dataset
#     weighted_data = data.withColumn(
#         "classWeightCol",
#         when(col(label_col) == 0, weights.get(0, 1.0))
#         .when(col(label_col) == 1, weights.get(1, 1.0))
#         .when(col(label_col) == 2, weights.get(2, 1.0))
#         .when(col(label_col) == 3, weights.get(3, 1.0))
#         .otherwise(1.0)
#     )

#     return weighted_data

def test_samples(model, experiment_name, run_name, mlflow):
    samples = [
        ("I absolutely love this product, it’s fantastic!", 1),
        ("Terrible customer service, I’m very disappointed.", 0),
        ("The item arrived as described, nothing more.", 2),
        ("What’s the weather like today?", 3),
        ("Best purchase I’ve made this year!", 1),
        ("I don’t hate it, but I wouldn’t buy it again.", 2),
        ("Broken when arrived. Waste of money!", 0),
        ("Can someone explain how this works?", 3),
        ("Decent quality for the price, meets expectations.", 2),
        ("Super fast delivery and excellent packaging!", 1)
    ]
    spark = SparkSession.builder \
        .appName("Text Classification") \
        .getOrCreate()
    sample_df = spark.createDataFrame([Row(Text=text, Label=label) for text, label in samples])
    # sample_df = clean_text_column(sample_df, input_col="Text", output_col="Text")
    prediction = model.transform(sample_df)
    
    # Log the predictions
    mlflow.log_text(prediction.select("Text", "Label", "prediction", "probability").toPandas().to_string(), "sample_predictions.txt")
    print("Sample predictions:")
    prediction.select("Text", "Label", "prediction", "probability").show(truncate=False)

def create_pipeline(use_hashing=False, stop_words=None):
    tokenizer = Tokenizer(inputCol="Text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stop_words)

    if use_hashing:
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="RawFeatures", numFeatures=10000)
        idf = IDF(inputCol="RawFeatures", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="Label", maxIter=20)
        # lr = LogisticRegression(featuresCol="features", labelCol="Label", weightCol="classWeightCol", maxIter=20)

        stages = [tokenizer, remover, hashingTF, idf, lr]
        pipeline = Pipeline(stages=stages)
        return pipeline, hashingTF, lr
    else:
        vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="Label", maxIter=20)
        # lr = LogisticRegression(featuresCol="features", labelCol="Label", weightCol="classWeightCol", maxIter=20)

        stages = [tokenizer, remover, vectorizer, lr]
        pipeline = Pipeline(stages=stages)
        return pipeline, vectorizer, lr



def tune_model(pipeline, train_data, use_hashing=True, vectorizer=None, hashingTF=None, lr=None):
    evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="f1")

    paramGrid = ParamGridBuilder()

    if use_hashing:
        paramGrid = paramGrid.addGrid(hashingTF.numFeatures, [1000, 5000, 10000])
    else:
        paramGrid = paramGrid.addGrid(vectorizer.vocabSize, [5000, 10000])

    paramGrid = paramGrid.addGrid(lr.regParam, [0.0, 0.01, 0.1])
    paramGrid = paramGrid.build()

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5
    )

    best_model = crossval.fit(train_data)
    return best_model


def evaluator(prediction1, label_col="Label", prediction_col="prediction"):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
    accuracy = evaluator.evaluate(prediction1, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(prediction1, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(prediction1, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(prediction1, {evaluator.metricName: "f1"})

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    return accuracy, precision, recall, f1

def data_distribution(data, label_col="Label"):
    total_count = data.count()
    label_dist = data.groupBy(label_col).count()
    label_dist = label_dist.withColumn("percentage", (col("count") / total_count) * 100)
    label_dist.orderBy(label_col).show()
    return label_dist.orderBy(label_col)

def load_config(config_path="configs/config.yaml"):       
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def set_up_mlflow_tracking(config, config_secret):
    """Sets up MLflow tracking URI and credentials."""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config_secret['mlflow']['password']
    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow']['tracking_uri']
    os.environ['MLFLOW_TRACKING_USERNAME'] = config['mlflow']['username']
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    logging.info("MLflow tracking setup complete.")
    
def load_dataset(config, config_secret):
    """Loads dataset from S3."""
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    logging.info(f"Input path for dataset: {dataset_path}")
    # AWS credentials and region
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    
    logging.info("Reading CSV file from S3...")
    data = read_csv_from_s3(dataset_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    logging.info("Read csv file from S3 successfully.")
    data.show(3)
    
    return data

def split_data(data, train_ratio=0.8, seed=42):
    train_data, validate_data = data.randomSplit([0.8, 0.2], seed=42)
    logging.info("Data split completed.")
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Validate data:")
    validate_data.printSchema() 
    validate_data.show(3)
    


def main(name="CountVectorizer_Model"):
    # ##################################################################################################
    # Configure logging within this process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting Text Classification Pipeline for {name}...")
    
    
    # Load configuration
    logging.info("Loading configuration from 'configs/config.yaml'")
    config = load_config("configs/config.yaml")
    config_secret = load_config("configs/secrets.yaml")
    logging.info("Successfully loaded configuration.")
    
    
    # Load dataset
    logging.info("Loading dataset from S3...")
    data = load_dataset(config, config_secret)
    data = data.filter(col("label") != 3)
    logging.info("Successfully loaded dataset.")
    
   

    # ##################################################################################################
    
    # Split data into train and validate sets
    logging.info("Splitting data into train and validate sets...")
    train_data, validate_data = data.randomSplit([0.9, 0.1], seed=42)
    # train_data = add_class_weights(train_data, label_col="Label")
    # validate_data = add_class_weights(validate_data, label_col="Label")

    logging.info("Data split completed.")
    
    
    # Number of samples
    total_count = data.count()
    logging.info(f"Total samples: {data.count()}")
    
    num_partitions = data.rdd.getNumPartitions()
    logging.info(f"Number of partitions: {num_partitions}")
    partitions = data.rdd.glom().collect()  # Glom giúp bạn nhìn thấy dữ liệu trong mỗi partition
    for i, partition in enumerate(partitions):
        logging.info(f"Partition {i} has {len(partition)} samples")
    

    # Distribution of labels of data
    print("Label distribution of dataset:")
    data_distribution(data, label_col="Label")
    
    print("Label distribution of train dataset:")
    data_distribution(train_data, label_col="Label")
    
    print("Label distribution of validate dataset:")
    data_distribution(validate_data, label_col="Label")
    
    
    
    logging.info("Creating pipeline...")
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')

    if name == "CountVectorizer_Model":
        pipeline, vectormethod, lr = create_pipeline(use_hashing=False, stop_words=stop_words)
    elif name == "HashingTF_IDF_Model":
        pipeline, vectormethod, lr = create_pipeline(use_hashing=True, stop_words=stop_words)
    # pipeline_cv, vectorizer_cv, lr_cv = create_pipeline(use_hashing=False, stop_words=stop_words)
    # pipeline_tf, hashingTF_tf, lr_tf = create_pipeline(use_hashing=True, stop_words=stop_words)

    logging.info("Pipeline created successfully.")
    
    
    # Set up MLflow experiment
    logging.info("Setting up MLflow experiment...")
    set_up_mlflow_tracking(config=config, config_secret=config_secret)
    experiment_name = f'{name}_Text_Classification_Experiment'
    mlflow.set_experiment(experiment_name)
    logging.info("MLflow experiment set up successfully with name: %s", experiment_name)

    
    logging.info(f"Tuning model {name} with CrossValidator...")
    with mlflow.start_run(run_name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") as run1:
        if name == "CountVectorizer_Model":
            model = tune_model(pipeline, train_data, use_hashing=False, vectorizer=vectormethod, lr=lr)
        elif name == "HashingTF_IDF_Model":
            model = tune_model(pipeline, train_data, use_hashing=True, hashingTF=vectormethod, lr=lr)
        logging.info(f"Model {name} tuning completed.")
        logging.info(f"Model {name} training completed.")
                
        # Log parameters
        if name == "CountVectorizer_Model":
            mlflow.log_param("vectorizer", "CountVectorizer")
            mlflow.log_param("hashing", False)
        elif name == "HashingTF_IDF_Model":
            mlflow.log_param("vectorizer", "HashingTF_IDF")
            mlflow.log_param("hashing", True)
        mlflow.log_param("stop_words", stop_words)

        
        # Evaluate the model
        prediction = model.transform(validate_data)
        accuracy, precision, recall, f1 = evaluator(prediction)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        
        # Log the model
        
        if name == "CountVectorizer_Model":
            mlflow.spark.log_model(model, "model_cv")
            model_uri = f"runs:/{run1.info.run_id}/model_cv"
            mlflow.register_model(model_uri, "CountVectorizer_Model")
            test_samples(model, experiment_name, "CountVectorizer_Model", mlflow)
        elif name == "HashingTF_IDF_Model":
            mlflow.spark.log_model(model, "model_tf")
            model_uri = f"runs:/{run1.info.run_id}/model_tf"
            mlflow.register_model(model_uri, "HashingTF_IDF_Model")
            test_samples(model, experiment_name, "HashingTF_IDF_Model", mlflow)
        
    logging.info("Evaluations completed and logged to MLflow.")
    


if __name__ == "__main__":
    process1 = Process(target=main, args=("CountVectorizer_Model",))
    process2 = Process(target=main, args=("HashingTF_IDF_Model",))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
 
    
    
    
    