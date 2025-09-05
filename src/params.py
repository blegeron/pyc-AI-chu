import os

MLFLOW_HOST = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")
MLFLOW_PORT = os.environ.get("MLFLOW_PORT", 5000)

MLFLOW_URI = f"{MLFLOW_HOST}:{MLFLOW_PORT}"

if __name__ == "__main__":
    print(f"MLFLOW_TRACKING_URI = {MLFLOW_URI}")
    print(f"MLFLOW_EXPERIMENT_NAME = {MLFLOW_EXPERIMENT_NAME}")
    print(f"MLFLOW_PORT = {MLFLOW_PORT}")
    print(f"MLFLOW_URI = {MLFLOW_URI}")
