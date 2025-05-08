import pandas as pd

# Đọc file CSV không có header
df = pd.read_csv("data/twitter_validation.csv", header=None)

# Gán header mới
df.columns = ["ID", "Product", "Label", "Text"]

# Hiển thị 5 dòng đầu tiên để kiểm tra
print(df.head())

df.to_csv("data/twitter_validation.csv", index=False)
