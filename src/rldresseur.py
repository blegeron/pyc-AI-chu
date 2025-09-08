import asyncio
import random
from collections import deque

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from loguru import logger
from poke_env import AccountConfiguration, MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.data import GenData

from params import MLFLOW_EXPERIMENT_NAME, MLFLOW_HOST
from registry import load_model, save_model


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SimpleRLAgent(Player):
    observation_space = Box(low=-1.0, high=4.0, shape=(10,), dtype=np.float32)

    def __init__(self, battle_format, q_net, buffer, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.q_net = q_net
        self.buffer = buffer
        self.epsilon = 1.0

    def embed_battle(self, battle):
        moves_base_power = -np.ones(4, dtype=np.float32)
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)
        type_chart = GenData.from_gen(9).type_chart  # For Gen 9 format

        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1, battle.opponent_active_pokemon.type_2, type_chart=type_chart
                )

        fainted_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [fainted_team, fainted_opponent]], axis=0
        ).astype(np.float32)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=30)

    def choose_move(self, battle):
        obs = torch.tensor(self.embed_battle(battle), dtype=torch.float32)
        if random.random() < self.epsilon:
            return self.choose_random_move(battle)
        with torch.no_grad():
            q = self.q_net(obs.unsqueeze(0))
        action = q.argmax(dim=1).item()
        return self._action_to_move(action, battle)

    def choose_switch(self, battle):
        obs = torch.tensor(self.embed_battle(battle), dtype=torch.float32)
        if random.random() < self.epsilon:
            return self.choose_random_move(battle)
        with torch.no_grad():
            q = self.q_net(obs.unsqueeze(0))
        action = q.argmax(dim=1).item()
        return self._action_to_move(action, battle)

    def _action_to_move(self, action, battle):
        options = battle.available_moves + battle.available_switches
        move = options[action % len(options)]
        return self.create_order(move)


def train_epoch(
    agent,
    target_net,
    buffer,
    optimizer,
    batch_size: int = 64,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 1e-4,
    update_target_every: int = 1000,
):
    mlflow.log_params({"gamma": 0.99, "batch_size": 64, "lr": 1e-3})
    # Decrease epsilon (linear decay)
    agent.epsilon = max(eps_end, agent.epsilon - eps_decay)

    # Skip training if buffer is too small
    if len(buffer) < batch_size:
        return

    # Sample a batch of transitions
    s, a, r, s2, done = buffer.sample(batch_size)
    # Predictions for current states
    q_values = agent.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Compute target values using target network
    with torch.no_grad():
        max_next_q = target_net(s2).max(1)[0]
        targets = r + gamma * (1 - done) * max_next_q

    # Compute loss and update network
    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Periodically sync target network
    train_step = getattr(agent, "train_step_count", 0) + 1
    agent.train_step_count = train_step
    if train_step % update_target_every == 0:
        target_net.load_state_dict(agent.q_net.state_dict())


def train_step(agent, target_net, buffer, optimizer, batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return

    s, a, r, s2, done = buffer.sample(batch_size)
    q_values = agent.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q = target_net(s2).max(1)[0]
        targets = r + gamma * (1 - done) * max_next_q

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


async def train_agent():
    for epoch in range(10):  # 10 epochs
        logger.info(f"Epoch {epoch + 1} started.")

        for battle_num in range(100):  # 100 battles per epoch
            logger.info(f"  Battle {battle_num + 1} started.")
            battle = await agent.battle_against(opponent, n_battles=1)
            # Perform training after each battle
            train_step(agent, target_net, buffer, optimizer)
            logger.info(f"  Battle {battle_num + 1} finished.")

        mlflow.log_metrics(
            {
                "lose": agent.n_lost_battles / 100,
                "win_rate": agent.n_won_battles / agent.n_finished_battles,
                "epsilon": agent.epsilon,
            },
            step=epoch,
        )

        if epoch % 2 == 0:
            mlflow.pytorch.log_model(
                pytorch_model=agent.q_net,
                name="dqn_model",
                input_example=np.zeros((1, 1, 10), dtype=np.float32),
                step=epoch,
            )

        logger.info(f"Epoch {epoch + 1} finished.")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_HOST)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    input_dim = 10
    output_dim = 26  # discrete number of possible moves
    # q_net = QNetwork(input_dim, output_dim)
    q_net = load_model("pokemon_torch_rl")
    target_net = QNetwork(input_dim, output_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net = target_net.cuda()
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10)

    account_configuration_agent = AccountConfiguration(
        username=f"SimpleRL{np.random.randint(1e5)}",
        password=None,
    )
    agent = SimpleRLAgent(
        account_configuration=account_configuration_agent, battle_format="gen9randombattle", q_net=q_net, buffer=buffer
    )
    account_configuration_oppo = AccountConfiguration(
        username=f"MaxBasePower{np.random.randint(1e5)}",
        password=None,
    )
    opponent = MaxBasePowerPlayer(account_configuration=account_configuration_oppo, battle_format="gen9randombattle")

    asyncio.run(train_agent())

    mlflow.log_metrics({"win_ratio": agent.n_won_battles / agent.n_finished_battles})
    logger.debug(f"Player {agent.username} won {agent.n_won_battles} out of {agent.n_finished_battles} played")

    mlflow.pytorch.log_model(
        pytorch_model=agent.q_net, name="dqn_model", input_example=np.zeros((1, 1, 10), dtype=np.float32)
    )
    save_model(agent.q_net, "pokemon_torch_rl")
