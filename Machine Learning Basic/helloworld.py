import subprocess

# 基本用法
result = subprocess.run(['ls'])

# 捕获输出
# result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
print(result)
# print(result.returncode)


















