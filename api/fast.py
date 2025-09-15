import os

import loguru
import numpy as np
from fastapi import FastAPI, HTTPException
from poke_env import AccountConfiguration, ServerConfiguration

from rldresseur import ReplayBuffer, SimpleRLAgent
from agenticdresseur import QwenAgent
from team_selector import generate_team
from utils.registry import get_all_models, load_model

my_api = FastAPI()


@my_api.get("/health")
def health():
    return {"status": "ok"}


@my_api.get("/agents")
def agents():
    return {"agents": get_all_models()}


@my_api.get("/duel")
async def duel(pseudo: str, agent: str, format: str = "gen9randombattle") -> None:
    custom_config = ServerConfiguration(
        "wss://showdown-899456062956.us-east1.run.app/showdown/websocket", "/action.php?"
    )
    loguru.logger.info(f"{pseudo} is trying dueling with {agent}!")
    try:
        account_configuration_agent = AccountConfiguration(
            username=f"{agent}{np.random.randint(1e2)}",
            password=None,
        )

        team = None
        if format != "gen9randombattle":
            team = generate_team(format=format)

        if "agentic" in agent:
            agent_battle = QwenAgent(
                account_configuration=account_configuration_agent,
                battle_format=format,
                team=team,
                server_configuration=custom_config,
            )
        else:
            buffer = ReplayBuffer(capacity=1000)

            # Replace with actual dims
            q_net = load_model(agent)
            agent_battle = SimpleRLAgent(
                account_configuration=account_configuration_agent,
                battle_format=format,
                team=team,
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
def get_team(format: str, constraint: str | None = None) -> dict:
    loguru.logger.info(f"Generating team for format {format} with constraint {constraint}")
    team = generate_team()
    return {"team": team}  # TODO use format and constraint
