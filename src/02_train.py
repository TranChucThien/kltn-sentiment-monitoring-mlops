import findspark
findspark.init()
from utils.s3_process import read_csv_from_s3
from utils.clean_text import clean_text_column
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
from pyspark.sql import Row
from pyspark.sql.functions import col
# 
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

# 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def test_samples(model):
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




if __name__ == "__main__":
    
    ##################################################################################################

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
    ##################################################################################################
    
    # Split data into train and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print("Train data:")
    train_data.printSchema()
    train_data.show(3)
    
    print("Test data:")
    test_data.printSchema()
    test_data.show(3)
    
    # train_data = data
    
    
    # Tổng số mẫu
    total_count = data.count()

    # Phân phối phần trăm
    label_dist = data.groupBy("Label").count()
    label_dist = label_dist.withColumn("percentage", (col("count") / total_count) * 100)
    label_dist.orderBy("Label").show()
    
        # Phân phối phần trăm
    total_count = train_data.count()
    label_dist = train_data.groupBy("Label").count()
    label_dist = label_dist.withColumn("percentage", (col("count") / total_count) * 100)
    label_dist.orderBy("Label").show()
    
        # Phân phối phần trăm
    total_count = test_data.count()
    label_dist = test_data.groupBy("Label").count()
    label_dist = label_dist.withColumn("percentage", (col("count") / total_count) * 100)
    label_dist.orderBy("Label").show()
    
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')

    pipeline_cv, vectorizer_cv, lr_cv = create_pipeline(use_hashing=False, stop_words=stop_words)
    pipeline_tf, hashingTF_tf, lr_tf = create_pipeline(use_hashing=True, stop_words=stop_words)

    model_cv = tune_model(pipeline_cv, train_data, use_hashing=False, vectorizer=vectorizer_cv, lr=lr_cv)
    model_tf = tune_model(pipeline_tf, train_data, use_hashing=True, hashingTF=hashingTF_tf, lr=lr_tf)

    
    # model_cv = pipeline_cv.fit(train_data)
    # model_tf = pipeline_tf.fit(train_data)
    
    prediction_cv = model_cv.transform(test_data)
    prediction_tf = model_tf.transform(test_data)
    
    prediction_cv.select("Text", "Label", "prediction", "probability").show(20, truncate=True)
    prediction_tf.select("Text", "Label", "prediction", "probability").show(20, truncate=True)
    

    print("Evaluation for CountVectorizer:")
    evaluator(prediction_cv)

    print("\nEvaluation for HashingTF + IDF:")
    evaluator(prediction_tf)

    print("\nSample predictions with CountVectorizer:")
    test_samples(model_cv)

    print("\nSample predictions with HashingTF + IDF:")
    test_samples(model_tf)