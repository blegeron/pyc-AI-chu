import os
import pickle

import torch
from loguru import logger

from q_network import QNetwork

dirname = os.path.dirname(__file__)
models_path = os.path.join(dirname, "models")

if not os.path.exists(models_path):
    os.makedirs(models_path)


def save_model(model: QNetwork, name: str) -> None:
    torch.save(model.state_dict(), f"{models_path}/{name}.pth")


def load_model(name: str, input_dim: int = 10, output_dim: int = 26) -> QNetwork:
    model = QNetwork(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(f"{models_path}/{name}.pth", map_location="cpu"))
    model.eval()
    return model


def get_all_models() -> list[str]:
    if not os.path.exists(models_path):
        return []
    dir_list = os.listdir(models_path)
    list_models = [f.split(".")[0] for f in dir_list if f.endswith(".pth")]
    return list_models


if __name__ == "__main__":
    print("Registry module")
