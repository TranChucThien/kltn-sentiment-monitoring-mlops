from setuptools import setup, find_packages
from os import path

# Đọc file requirements.txt
def read_requirements():
    with open(path.join(path.dirname(__file__), 'requirements.txt')) as f:
        return f.read().splitlines()

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),  # Cài đặt các dependencies từ requirements.txt
)
