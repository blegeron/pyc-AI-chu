import os

import loguru
import numpy as np
from fastapi import FastAPI, HTTPException
from poke_env import AccountConfiguration, ServerConfiguration

from registry import load_model
from rldresseur import ReplayBuffer, SimpleRLAgent

my_api = FastAPI()


@my_api.get("/health")
def health():
    return {"status": "ok"}


@my_api.get("/agents")
def agents():
    dirname = os.path.dirname(__file__)
    models_path = os.path.join(dirname, "../src/models")
    dir_list = os.listdir(models_path)
    if not os.path.exists(models_path):
        return {"agents": []}
    list_agents = [f.split(".")[0] for f in dir_list]
    return {"agents": list_agents}


@my_api.get("/duel")
async def duel(pseudo: str, agent: str) -> None:
    custom_config = ServerConfiguration(
        "wss://showdown-899456062956.us-east1.run.app/showdown/websocket", "/action.php?"
    )
    loguru.logger.info(f"{pseudo} is trying dueling with {agent}!")
    try:
        buffer = ReplayBuffer(capacity=1000)

        # Replace with actual dims
        q_net = load_model(agent)
        account_configuration_agent = AccountConfiguration(
            username=f"{agent}{np.random.randint(1e2)}",
            password=None,
        )
        agent_battle = SimpleRLAgent(
            account_configuration=account_configuration_agent,
            battle_format="gen9randombattle",
            q_net=q_net,
            buffer=buffer,
            server_configuration=custom_config,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Agent model file not found")
    except Exception as e:
        loguru.logger.exception("Unexpected error while loading agent")
        raise HTTPException(status_code=500, detail=str(e))

    loguru.logger.info(f"{pseudo} is dueling with {agent}!")
    await agent_battle.send_challenges(pseudo, n_challenges=1)
    return {"status": "ok"}


@my_api.get("/team")
def get_team(format: str, constraint: str | None = None) -> str:
    loguru.logger.info(f"Generating team for format {format} with constraint {constraint}")
    pass
