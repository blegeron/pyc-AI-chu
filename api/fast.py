import os

import loguru
from fastapi import FastAPI, HTTPException

from registry import load_model
from rldresseur import ReplayBuffer, SimpleRLAgent

my_api = FastAPI()


@my_api.get("/health")
def health():
    return {"status": "ok"}


@my_api.get("/agents")
def agents():
    path = "C://Users//Vanshi//Desktop//gfg"
    dir_list = os.listdir(path)
    list_agents = [f.split(".")[0] for f in dir_list]
    return {"agents": list_agents}


@my_api.get("/duel")
def duel(pseudo: str, agent: str) -> None:
    try:
        load_model(agent)
    except HTTPException:
        return HTTPException(status_code=404, detail="Agent not Found check agent with /agents endpoint")
    loguru.logger.info(f"{pseudo} is dueling with {agent}!")
    buffer = ReplayBuffer(capacity=1)
    agent_test = SimpleRLAgent(battle_format="gen9randombattle", q_net=load_model(agent), buffer=buffer)
    await agent_test.send_challenges(pseudo, n_challenges=1)
    return {"status": "ok"}


@my_api.get("/team")
def get_team(format: str, constraint: str | None = None) -> str:
    loguru.logger.info(f"Generating team for format {format} with constraint {constraint}")
    pass
