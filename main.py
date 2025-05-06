# how to run this file
# python pipeline.py all        # Chạy toàn bộ pipeline
# python pipeline.py dataset    # Chỉ chuẩn bị dữ liệu
# python pipeline.py train      # Chỉ train
# python pipeline.py eval       # Chỉ evaluate

import argparse
from src._01_dataset_preparation import main as dataset_main
from src._02_training import main as training_main
from src._03_test import main as test_main
import threading

def run_dataset():
    print("📦 Running Dataset Preparation...")
    dataset_main()

def run_training():
    print("🧠 Running Training...")
    thread1 = threading.Thread(target=run_training_count_vector)
    thread2 = threading.Thread(target=run_training_hashing_tf)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

def run_training_count_vector():
    print("🧠 Running Training for CountVectorizer Model...")
    training_main(name="CountVectorizer_Model")

def run_training_hashing_tf():
    print("🧠 Running Training for HashingTF Model...")
    training_main(name="HashingTF_IDF_Model")

def run_test_count_vector():
    print("📊 Running Test for CountVecorizer Model...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    test_main(model_name="CountVectorizer_Model", use_latest_version=True)
    # evaluate_main(name="HashingTF_IDF_Model", input_data=False)

def run_test_hashing_tf():
    print("📊 Running Test for HashingTF Model...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    test_main(model_name="HashingTF_IDF_Model", use_latest_version=True)
def run_test():
    print("📊 Running Test...")
    # cv_version, tf_version = training_main()  # hoặc load version từ file
    thread1 = threading.Thread(target=run_test_count_vector)
    thread2 = threading.Thread(target=run_test_hashing_tf)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("step", choices=["all", "dataset", "train", "train_count_vector", "train_hashing_tf", "test", "test_count_vector", "test_hashing_tf"],
                        help="Which step to run")

    args = parser.parse_args()

    if args.step == "all":
        run_dataset()
        run_training()
        thread1 = threading.Thread(target=run_test_count_vector)
        thread2 = threading.Thread(target=run_test_hashing_tf)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        
    elif args.step == "dataset":
        run_dataset()
    elif args.step == "train":
        run_training()
    elif args.step == "test_count_vector":
        run_test_count_vector()
    elif args.step == "test_hashing_tf":
        run_test_hashing_tf()
    elif args.step == "test":
        run_test()
    elif args.step == "train_count_vector":
        run_training_count_vector()
    elif args.step == "train_hashing_tf":
        run_training_hashing_tf()

if __name__ == "__main__":
    main()

