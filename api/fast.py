import loguru
from fastapi import FastAPI

my_api = FastAPI()


@my_api.get("/health")
def health():
    return {"status": "ok"}


@my_api.get("/duel")
def duel(pseudo: str, agent: str) -> None:
    loguru.logger.info(f"{pseudo} is dueling with {agent}!")
    pass


@my_api.get("/team")
def get_team(format: str, constraint: str | None = None) -> str:
    loguru.logger.info(f"Generating team for format {format} with constraint {constraint}")
    pass
