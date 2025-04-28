from src._01_dataset_preparation import main as preprocessing_main
from src._02_training import main as train_main
from src._03_evaluate import main as evaluate_main

preprocessing_main()
cv_version, tf_version = train_main()
evaluate_main(name="CountVectorizer_Model", version=cv_version)
evaluate_main(name="HashingTF_IDF_Model", version=tf_version)