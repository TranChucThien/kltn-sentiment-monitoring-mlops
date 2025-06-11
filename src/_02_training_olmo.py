import findspark
findspark.init()
from utils.s3_process import  read_key
from utils.s3_process_nlp import read_csv_from_s3, get_latest_s3_object_version
from utils.clean_text import clean_text_column

from utils.mlflow_func import get_latest_model_version, get_model_version_by_stage
import datetime
import yaml
import mlflow
import os 
import time

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
import logging

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType




def create_pipeline():
    
    document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
        
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized") \
        .setLowercase(True) 
        
    stop_words_cleaner = StopWordsCleaner() \
        .setInputCols(["normalized"]) \
        .setOutputCol("cleanTokens") \
        .setCaseSensitive(False)
    
    lemmatizer = LemmatizerModel.pretrained("lemma_antbnc") \
        .setInputCols(["cleanTokens"]) \
        .setOutputCol("lemmatized") 
            
    word_embeddings_elmo = ElmoEmbeddings.pretrained("elmo", "en") \
        .setInputCols(["document", "lemmatized"]) \
        .setOutputCol("embeddings")

    sentence_embeddings = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")  
        
    classifier = ClassifierDLApproach() \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("category") \
        .setLabelColumn("label") \
        .setMaxEpochs(15) \
        .setLr(0.003) \
        .setBatchSize(8) \
        .setEnableOutputLogs(True) \
        .setOutputLogsPath("classifier_logs")
    
    finisher = Finisher() \
        .setInputCols(["category"]) \
        .setOutputCols(["prediction"]) \
        .setCleanAnnotations(False) \
        
        
    pipeline_elmo = Pipeline(stages=[
        document_assembler,
        tokenizer,
        normalizer,
        stop_words_cleaner,
        lemmatizer,
        word_embeddings_elmo,
        sentence_embeddings,
        classifier,
        finisher
    ])
    return pipeline_elmo

def k_fold_split(data, k=3, seed=42):
    # Chia dữ liệu thành k phần bằng randomSplit
    weights = [1.0 / k] * k
    return data.randomSplit(weights, seed=seed)



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


def evaluator(prediction1, label_col="label", prediction_col="prediction"):
    prediction1 = prediction1.withColumn(label_col, col(label_col).cast(DoubleType()))
    prediction1 = prediction1.withColumn(prediction_col, col(prediction_col).cast(DoubleType()))
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
    
def load_dataset(config, config_secret, spark):
    """Loads dataset from S3."""
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    logging.info(f"Dataset path: {dataset_path}")
    # AWS credentials and region
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    
    logging.info("Reading CSV file from S3...")
    data = read_csv_from_s3(dataset_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, spark)
    logging.info("Read csv file from S3 successfully.")
    data.show(3)
    
    
    return data


def get_data_version(config, config_secret):
    """Loads dataset from S3."""
    bucket = config['s3']['bucket']
    dataset_key = config['s3']['keys']['dataset']
    
    dataset_path = f"s3a://{bucket}/{dataset_key}"
    logging.info(f"Dataset path: {dataset_path}")
    # AWS credentials and region
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_key(config_secret['aws']['access_key_path'])
    AWS_REGION = config['aws']['region']
    S3_OUTPUT_KEY = config['s3']['keys']['dataset']
    BUCKET_NAME = config['s3']['bucket']
    
    return get_latest_s3_object_version(s3_path=dataset_path,aws_access_key=AWS_ACCESS_KEY_ID,aws_secret_key=AWS_SECRET_ACCESS_KEY,region=AWS_REGION), dataset_path

def split_data(data, train_ratio=0.8, seed=42):
    train_data, validate_data = data.randomSplit([0.8, 0.2], seed=42)
    logging.info("Data split completed.")
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Validate data:")
    validate_data.printSchema()
    validate_data.show(3)
    


def main():
    # ##################################################################################################
    # Configure logging within this process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting Text Classification Pipeline for Olmo model...")
    
    
    # Load configuration
    logging.info("Loading configuration from 'configs/config.yaml'")
    config = load_config("configs/config.yaml")
    config_secret = load_config("configs/secrets.yaml")
    logging.info("Successfully loaded configuration.")
    
    # Spark session initialization
    logging.info("Initializing Spark session...")
    spark = sparknlp.start(
        SparkSession.builder \
            .appName("Spark NLP - BERT Sentiment Classification") \
            .config("spark.driver.memory", "4G") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3") \
            .config("spark.kryoserializer.buffer.max", "2000M") \
            .config("spark.driver.maxResultSize", "0") \
            .getOrCreate()
    )
    logging.info("Spark Session with Spark NLP is ready.")
    
    # Load dataset
    logging.info("Loading dataset from S3...")
    data = load_dataset(config, config_secret, spark)
    data_version, dataset_path = get_data_version(config, config_secret)
    
    data = data.withColumnRenamed("text", "text").withColumnRenamed("label", "label")
    data = data.selectExpr("cast(Text as string) as text", "cast(Label as string) as label")
    data = data.filter(col("label") != "3")
    logging.info("Successfully loaded dataset.")
    
   

    # ##################################################################################################
    
    # Split data into train and validate sets
    logging.info("Splitting data into train and validate sets...")
    # data1, data2, data3, data4, data5 = data.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1 ], seed=42)
    train_data, validate_data = data.randomSplit([0.8, 0.2], seed=42)
    logging.info("Data split completed.")
    
    
    # Number of samples
    total_count = data.count()
    logging.info(f"Total samples: {data.count()}")

    # # Distribution of labels of data
    # print("Label distribution of dataset:")
    # data_distribution(data, label_col="Label")
    
    # print("Label distribution of train dataset:")
    # data_distribution(train_data, label_col="Label")
    
    # print("Label distribution of validate dataset:")
    # data_distribution(validate_data, label_col="Label")
    
    
    
    logging.info("Creating pipeline...")
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # stop_words = stopwords.words('english')

    
    pipeline = create_pipeline()

    logging.info("Pipeline created successfully.")
    
    
    # Set up MLflow experiment
    logging.info("Setting up MLflow experiment...")
    set_up_mlflow_tracking(config=config, config_secret=config_secret)
    experiment_name = f'DL_Elmo_Text_Classification_Experiment'
    mlflow.set_experiment(experiment_name)
    logging.info("MLflow experiment set up successfully with name: %s", experiment_name)

    
    
    with mlflow.start_run(run_name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") as run1:

        start = time.time()
        logging.info("Starting training of DL Elmo model at %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        model_elmo = pipeline.fit(data)
        end = time.time()
        logging.info("Training completed at %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logging.info(f"Model DL Elmo training completed.")
        
               
        # Log parameters
        mlflow.log_param("embedding_type", "elmo")
        mlflow.log_param("pooling_strategy", "AVERAGE")
        mlflow.log_param("epochs", 10)
        mlflow.log_param("learning_rate", 0.003)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("lemmatizer", "lemma_antbnc")

        # log dataset information
        mlflow.log_param("total_dataset_samples", total_count)
        mlflow.log_param("total_train_samples", train_data.count())
        mlflow.log_param("total_validate_samples", validate_data.count())
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("dataset_path", dataset_path)
        logging.info("Model parameters logged to MLflow.")
        
        # log training time
        mlflow.log_metric("training_time_minutue", (end - start) / 60)
        logging.info(f"Model training time: {(end - start) / 60:.2f} minutes")
        
        # Evaluate the model
        prediction = model_elmo.transform(validate_data).withColumn("prediction", col("prediction")[0].cast("string"))
        accuracy, precision, recall, f1 = evaluator(prediction, label_col="label", prediction_col="prediction")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log the model
        mlflow.spark.log_model(model_elmo, "model_elmo")
        model_uri = f"runs:/{run1.info.run_id}/model_elmo"

        mlflow.register_model(model_uri, "DL_Elmo_Text_Classification_Model")
        logging.info("Model logged to MLflow with URI: %s", model_uri)
        

        
    logging.info("Evaluations completed and logged to MLflow.")
    


if __name__ == "__main__":
    main()


 
    
    
    
    