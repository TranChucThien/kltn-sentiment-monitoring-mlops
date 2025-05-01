from src._01_dataset_preparation import main as _01_dataset_preparation_main
from src._02_training import main as _02_training_main
from src._03_evaluate import main as _03_evaluate_main

print("🚀 Running the entire pipeline...")
print("📦 Running Dataset Preparation...")
_01_dataset_preparation_main()
print("🧠 Running Training...")
cv_version, tf_version = _02_training_main()
print(f"✅ Training done: CV v{cv_version}, TF v{tf_version}")
print("📊 Running Evaluation...")
_03_evaluate_main(name="CountVectorizer_Model", version=cv_version)
_03_evaluate_main(name="HashingTF_IDF_Model", version=tf_version)
print("✅ Evaluation done.")
print("🚀 Pipeline completed successfully!")