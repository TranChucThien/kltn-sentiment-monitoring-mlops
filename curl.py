import subprocess
import json
import argparse

# Cấu hình argparse để nhận tham số run_option từ dòng lệnh
parser = argparse.ArgumentParser(description='Trigger GitHub workflow dispatch')
parser.add_argument('--run_option', required=True, choices=['all', 'one'], help='Run option for the workflow')

args = parser.parse_args()

# Tham số run_option
run_option = args.run_option

# Các thông tin khác (cần thay đổi theo nhu cầu của bạn)
token = "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"  # Thay thế bằng GitHub Personal Access Token của bạn
owner = "TranChucThien"  # Tên người dùng GitHub hoặc tổ chức
repo = "argocd-example"  # Tên repository của bạn
workflow_file_name = "blank.yml"  # Đường dẫn đến workflow
ref = "main"  # Tên nhánh bạn muốn trigger

# URL API GitHub để gọi workflow dispatch
url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file_name}/dispatches"

# Payload data với input 'run_option'
data = {
    "ref": ref,  # Tên nhánh bạn muốn trigger
    "inputs": {
        "run_option": run_option
    }
}

# Gọi API sử dụng curl
command = [
    "curl",
    "-X", "POST",
    "-H", f"Authorization: token {token}",
    "-H", "Accept: application/vnd.github.v3+json",
    "-d", json.dumps(data),
    url
]

# Thực thi command
subprocess.run(command)
