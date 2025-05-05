import findspark
findspark.init()
from utils.s3_process import read_csv_from_s3, read_key
from utils.clean_text import clean_text_column
from utils.logger import setup_logger
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
from pyspark.sql.functions import col
# 
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

# 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
from multiprocessing import Process

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
        stages = [tokenizer, remover, hashingTF, idf, lr]
        pipeline = Pipeline(stages=stages)
        return pipeline, hashingTF, lr
    else:
        vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="Label", maxIter=20)
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
        numFolds=3
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

def main(name="CountVectorizer_Model"):
    # ##################################################################################################
    logger = setup_logger("TextClassificationPipeline")

    # Load configuration
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded configuration.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    print("Input path:", dataset_path)
    # AWS credentials and region
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config['aws']['access_key_path'])
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    os.environ["MLFLOW_TRACKING_PASSWORD"]= config['mlflow']['password']    
    logger.info("Reading CSV file from S3...")
    data = read_csv_from_s3(dataset_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    logger.info("Read csv file from S3 successfully.")
    data.show(3)
    # ##################################################################################################
    
    # Split data into train and validate sets
    logger.info("Splitting data into train and validate sets...")
    train_data, validate_data = data.randomSplit([0.8, 0.2], seed=42)
    logger.info("Data split completed.")
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Validate data:")
    validate_data.printSchema()
    validate_data.show(3)
    
    
    # train_data = data
    
    
    # Tổng số mẫu

    total_count = data.count()
    logger.info(f"Total samples: {data.count()}")

    # Distribution of labels of data
    print("Label distribution of dataset:")
    data_distribution(data, label_col="Label")
    
    print("Label distribution of train dataset:")
    data_distribution(train_data, label_col="Label")
    
    print("Label distribution of validate dataset:")
    data_distribution(validate_data, label_col="Label")
    
    
    
    logger.info("Creating pipeline...")
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')

    if name == "CountVectorizer_Model":
        pipeline, vectormethod, lr = create_pipeline(use_hashing=False, stop_words=stop_words)
    elif name == "HashingTF_IDF_Model":
        pipeline, vectormethod, lr = create_pipeline(use_hashing=True, stop_words=stop_words)
    # pipeline_cv, vectorizer_cv, lr_cv = create_pipeline(use_hashing=False, stop_words=stop_words)
    # pipeline_tf, hashingTF_tf, lr_tf = create_pipeline(use_hashing=True, stop_words=stop_words)

    logger.info("Pipeline created successfully.")
    
       
    # Set up MLflow experiment
    experiment_name = f'Text_Classification_Experiment_{name}'
    mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow")  # Update this if you have a remote server
    
    logger.info(f"Tuning model {name} with CrossValidator...")
    with mlflow.start_run(run_name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") as run1:
        if name == "CountVectorizer_Model":
            model = tune_model(pipeline, train_data, use_hashing=False, vectorizer=vectormethod, lr=lr)
        elif name == "HashingTF_IDF_Model":
            model = tune_model(pipeline, train_data, use_hashing=True, hashingTF=vectormethod, lr=lr)
        logger.info("Model tuning completed.")
        logger.info("Model training completed.")
                
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
        
    logger.info("Evaluations completed and logged to MLflow.")
    
# def main_old():
    # ##################################################################################################
    logger = setup_logger("TextClassificationPipeline")

    # Load configuration
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded configuration.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    print("Input path:", dataset_path)
    # AWS credentials and region
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config['aws']['access_key_path'])
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    os.environ["MLFLOW_TRACKING_PASSWORD"]= config['mlflow']['password']    
    logger.info("Reading CSV file from S3...")
    data = read_csv_from_s3(dataset_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    logger.info("Read csv file from S3 successfully.")
    data.show(3)
    # ##################################################################################################
    
    # Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    logger.info("Data split completed.")
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Test data:")
    test_data.printSchema()
    test_data.show(3)
    
    
    # train_data = data
    
    
    # Tổng số mẫu

    total_count = data.count()
    logger.info(f"Total samples: {data.count()}")

    # Distribution of labels of data
    print("Label distribution of dataset:")
    data_distribution(data, label_col="Label")
    
    print("Label distribution of train dataset:")
    data_distribution(train_data, label_col="Label")
    
    print("Label distribution of test dataset:")
    data_distribution(test_data, label_col="Label")
    
    
    
    logger.info("Creating pipeline...")
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')

    pipeline_cv, vectorizer_cv, lr_cv = create_pipeline(use_hashing=False, stop_words=stop_words)
    pipeline_tf, hashingTF_tf, lr_tf = create_pipeline(use_hashing=True, stop_words=stop_words)

    logger.info("Pipeline created successfully.")
    
    
    
    experiment_name = f"Text_Classification_Experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
    mlflow.set_experiment(experiment_name)

    mlflow.set_tracking_uri("https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow")  # Update this if you have a remote server
    
    logger.info("Tuning model with CrossValidator...")
    with mlflow.start_run(run_name="CountVectorizer_Model") as run1:
        model_cv = tune_model(pipeline_cv, train_data, use_hashing=False, vectorizer=vectorizer_cv, lr=lr_cv)
        logger.info("Model tuning completed.")
        logger.info("Model training completed.")
        
        # Log parameters
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("hashing", False)
        mlflow.log_param("stop_words", stop_words)
        # for param_set in param_map_cv:
        #     mlflow.log_param(f"max_vocab_size", param_set[vectorizer_cv.vocabSize])
        #     mlflow.log_param(f"reg_param", param_set[lr_cv.regParam])
        
        # Evaluate the model
        prediction_cv = model_cv.transform(test_data)
        accuracy_cv, precision_cv, recall_cv, f1_cv = evaluator(prediction_cv)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_cv)
        mlflow.log_metric("precision", precision_cv)
        mlflow.log_metric("recall", recall_cv)
        mlflow.log_metric("f1", f1_cv)
        
        # Log the model
        mlflow.spark.log_model(model_cv, "model_cv")
        model_uri = f"runs:/{run1.info.run_id}/model_cv"
        mlflow.register_model(model_uri, "CountVectorizer_Model")
        
        test_samples(model_cv, experiment_name, "CountVectorizer_Model", mlflow)
        


    with mlflow.start_run(run_name="HashingTF_IDF_Model") as run2:
        model_tf = tune_model(pipeline_tf, train_data, use_hashing=True, hashingTF=hashingTF_tf, lr=lr_tf)
        logger.info("Model tuning completed.")
        logger.info("Model training completed.")
        
        # Log parameters
        mlflow.log_param("vectorizer", "HashingTF_IDF")
        mlflow.log_param("hashing", True)
        mlflow.log_param("stop_words", stop_words)
        # for param_set in param_map_tf:
        #     mlflow.log_param(f"num_features", param_set[hashingTF_tf.numFeatures])
        #     mlflow.log_param(f"reg_param", param_set[lr_tf.regParam])
            
        # Evaluate the model
        prediction_tf = model_tf.transform(test_data)
        accuracy_tf, precision_tf, recall_tf, f1_tf = evaluator(prediction_tf)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_tf)
        mlflow.log_metric("precision", precision_tf)
        mlflow.log_metric("recall", recall_tf)
        mlflow.log_metric("f1", f1_tf)
        
        # Log the model
        mlflow.spark.log_model(model_tf, "model_tf")
        model_uri = f"runs:/{run2.info.run_id}/model_tf"
        mlflow.register_model(model_uri, "HashingTF_IDF_Model")
        test_samples(model_tf, experiment_name, "HashingTF_IDF_Model",mlflow)
        
    logger.info("Evaluations completed and logged to MLflow.")
    return get_latest_model_version("CountVectorizer_Model").version, get_latest_model_version("HashingTF_IDF_Model").version

if __name__ == "__main__":
    process1 = Process(target=main, args=("CountVectorizer_Model",))
    process2 = Process(target=main, args=("HashingTF_IDF_Model",))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
 
    
    
    
    