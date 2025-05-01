# how to run this file
# python pipeline.py all        # Chạy toàn bộ pipeline
# python pipeline.py dataset    # Chỉ chuẩn bị dữ liệu
# python pipeline.py train      # Chỉ train
# python pipeline.py eval       # Chỉ evaluate

import argparse
from src._01_dataset_preparation import main as dataset_main
from src._02_training import main as training_main
from src._03_evaluate import main as evaluate_main

def run_dataset():
    print("📦 Running Dataset Preparation...")
    dataset_main()

def run_training():
    print("🧠 Running Training...")
    cv_version, tf_version = training_main()
    print(f"✅ Training done: CV v{cv_version}, TF v{tf_version}")

def run_evaluate():
    print("📊 Running Evaluation...")
    cv_version, tf_version = training_main()  # hoặc load version từ file
    evaluate_main(name="CountVectorizer_Model", version=cv_version)
    evaluate_main(name="HashingTF_IDF_Model", version=tf_version)

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("step", choices=["all", "dataset", "train", "eval"],
                        help="Which step to run")

    args = parser.parse_args()

    if args.step == "all":
        run_dataset()
        run_training()
        run_evaluate()
    elif args.step == "dataset":
        run_dataset()
    elif args.step == "train":
        run_training()
    elif args.step == "eval":
        run_evaluate()

        

if __name__ == "__main__":
    main()

