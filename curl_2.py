import subprocess
import json
import argparse
import os


def curl(run_option:str, clean_infra='false', provision_infra='true'):
    """
    Function to send a POST request to the GitHub Actions API to trigger a workflow.
    
    :param run_option: Run option for the pipeline
    :param clean_infra: Whether to clean the infrastructure or not
    :param provision_infra: Whether to provision the infrastructure or not
    """
    # Lấy token từ biến môi trường
    token = os.getenv("GITHUB_TOEKN")
    if not token:
        raise ValueError("You need to set the GITHUB_TOKEN environment variable")

    owner = "TranChucThien"
    repo = "kltn-sentiment-monitoring-mlops"
    workflow_file_name = "MLOPS_Pipeline_v3.yml"
    ref = "main"

    # URL API
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file_name}/dispatches"

    # Dữ liệu gửi đi
    data = {
        "ref": ref,
        "inputs": {
            "run_option": run_option,
            "clean_infra": clean_infra,
            "provision_infra": provision_infra,
        }
    }

    # Gọi API qua curl
    command = [
        "curl",
        "-X", "POST",
        "-H", f"Authorization: token {token}",
        "-H", "Accept: application/vnd.github.v3+json",
        "-d", json.dumps(data),
        url
    ]

    # Chạy lệnh
    subprocess.run(command)

curl("test", clean_infra='false', provision_infra='true')