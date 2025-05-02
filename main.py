# how to run this file
# python pipeline.py all        # Chạy toàn bộ pipeline
# python pipeline.py dataset    # Chỉ chuẩn bị dữ liệu
# python pipeline.py train      # Chỉ train
# python pipeline.py eval       # Chỉ evaluate

import argparse
from src._01_dataset_preparation import main as dataset_main
from src._02_training import main as training_main
from src._03_evaluate import main as evaluate_main
import threading

def run_dataset():
    print("📦 Running Dataset Preparation...")
    dataset_main()

def run_training():
    print("🧠 Running Training...")
    cv_version, tf_version = training_main()
    print(f"✅ Training done: CV v{cv_version}, TF v{tf_version}")

def run_evaluate_count_vector():
    print("📊 Running Evaluation for CountVecorizer Model...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    evaluate_main(name="CountVectorizer_Model", input_data=False)
    # evaluate_main(name="HashingTF_IDF_Model", input_data=False)

def run_evaluate_hashing_tf():
    print("📊 Running Evaluation for HashingTF Model...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    evaluate_main(name="HashingTF_IDF_Model", input_data=False)
def run_evaluate():
    print("📊 Running Evaluation...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    run_evaluate_count_vector()
    run_evaluate_hashing_tf()
    thread1 = threading.Thread(target=run_evaluate_count_vector)
    thread2 = threading.Thread(target=run_evaluate_hashing_tf)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("step", choices=["all", "dataset", "train", "eval", "eval_count_vector", "eval_hashing_tf"],
                        help="Which step to run")

    args = parser.parse_args()

    if args.step == "all":
        run_dataset()
        run_training()
        thread1 = threading.Thread(target=run_evaluate_count_vector)
        thread2 = threading.Thread(target=run_evaluate_hashing_tf)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
    elif args.step == "dataset":
        run_dataset()
    elif args.step == "train":
        run_training()
    elif args.step == "eval_count_vector":
        run_evaluate_count_vector()
    elif args.step == "eval_hashing_tf":
        run_evaluate_hashing_tf()
    elif args.step == "eval":
        run_evaluate()
        

if __name__ == "__main__":
    main()

