import os
import pickle

import pandas as pd
import torch
from loguru import logger

from models.q_network import QNetwork

dirname = os.path.dirname(__file__)
models_path = os.path.join(dirname, "../models")
data_path = os.path.join(models_path, "data")
saved_models_path = os.path.join(models_path, "saved")

if not os.path.exists(models_path):
    os.makedirs(models_path)


def save_team(smogon_team: str, name: str = "default") -> None:
    with open(f"{data_path}/{name}_team.txt", "wb") as f:
        f.write(smogon_team.encode("utf-8"))
    logger.info(f"Team saved at {data_path}/{name}_team.pkl")


def load_team(name: str = "default") -> list[str]:
    with open(f"{data_path}/{name}_team.pkl", "rb") as f:
        team = pickle.load(f)
    return team


def save_model(model: QNetwork, name: str) -> None:
    torch.save(model.state_dict(), f"{saved_models_path}/{name}.pth")


def load_model(name: str, input_dim: int = 10, output_dim: int = 26) -> QNetwork:
    model = QNetwork(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(f"{saved_models_path}/{name}.pth", map_location="cpu"))
    model.eval()
    return model


def get_all_models() -> list[str]:
    if not os.path.exists(saved_models_path):
        return []
    dir_list = os.listdir(saved_models_path)
    list_models = [f.split(".")[0] for f in dir_list if f.endswith(".pth")]
    return list_models


def pokemon_data() -> pd.DataFrame:
    return pd.read_csv(data_path + "/pokemons_selector.csv")


if __name__ == "__main__":
    print("Registry module")
