import os
import pickle

if not os.path.exists("models"):
    os.makedirs("models")


def save_model(model, name: str) -> None:
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(name: str):
    with open(f"models/{name}.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Registry module")
