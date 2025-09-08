import os
import pickle

dirname = os.path.dirname(__file__)
models_path = os.path.join(dirname, "models")

if not os.path.exists(models_path):
    os.makedirs(models_path)


def save_model(model, name: str) -> None:
    with open(f"{models_path}/{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(name: str):
    with open(f"{models_path}/{name}.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Registry module")
