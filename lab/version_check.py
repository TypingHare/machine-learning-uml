from importlib.metadata import version
import sys
from sys import version as python_formatted_version

sn = input("student name: ")
print(f"name: {sn}")
print(f"Python version: {sys.version_info[0]}.{sys.version_info[1]}")
print(python_formatted_version)
List = ["numpy", "scipy", "matplotlib", "jupyterlab"]
for item in List:
    print(f"module name: {item}, version: {version(item)}")
