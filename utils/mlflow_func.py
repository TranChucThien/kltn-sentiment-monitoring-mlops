from mlflow.tracking import MlflowClient
import os
def get_all_model_versions(model_name: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return versions

def get_latest_model_version(model_name: str):
    versions = get_all_model_versions(model_name)
    if not versions:
        return None
    latest_version = max(versions, key=lambda v: v.last_updated_timestamp)
    return latest_version

def get_model_version_by_stage(model_name: str, stage: str = "Production"):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
    if not latest_versions:
        return None
    return latest_versions[0]

def main():
    os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow"
    os.environ['MLFLOW_TRACKING_USERNAME']="TranChucThien"
    os.environ['MLFLOW_TRACKING_PASSWORD']="fc6e085ac3b8abf78e838bec58a25872e3db8679"
    # mlflow.set_tracking_uri("https://dagshub.com/TranChucThien/kltn-sentiment-monitoring-mlops.mlflow")  # Update this if you have a remote server

    model_name = "CountVectorizer_Model"  # Thay tên này bằng mô hình bạn đã đăng ký trên MLflow

    print("📦 All versions:")
    all_versions = get_all_model_versions(model_name)
    if all_versions:
        for v in all_versions:
            print(f" - Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")
    else:
        print("No versions found.")

    print("\n🕓 Latest version by last_updated_timestamp:")
    latest = get_latest_model_version(model_name)
    if latest:
        print(f" - Version: {latest.version}, Updated at: {latest.last_updated_timestamp}")
    else:
        print("No latest version found.")

    print("\n🚀 Version in Production stage:")
    prod_version = get_model_version_by_stage(model_name, "Production")
    if prod_version:
        print(f" - Version: {prod_version.version}, Run ID: {prod_version.run_id}")
    else:
        print("No model found in 'Production' stage.")

if __name__ == "__main__":
    main()
