import asyncio
import random
from collections import deque

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from loguru import logger
from poke_env import AccountConfiguration, MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.data import GenData

from params import MLFLOW_EXPERIMENT_NAME, MLFLOW_HOST
from q_network import QNetwork
from registry import load_model, save_model


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

    def __init__(self, battle_format: AbstractBattle, q_net: QNetwork, buffer: ReplayBuffer, device=None, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.q_net = q_net
        self.buffer = buffer
        self.epsilon = 1.0
        self.device = device if device is not None else torch.device("cpu")

        # Store previous state and action for experience replay
        self.prev_state = None
        self.prev_action = None
        self.prev_battle_id = None

    def embed_battle(self, battle: AbstractBattle):
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

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.15,
        hp_value: float = 0.15,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.15,
        victory_value: float = 1.0,
    ) -> float:
        """Compute reward based on battle state with configurable parameters

        Args:
            battle: The current battle state
            fainted_value: Weight for fainted pokemon difference
            hp_value: Weight for HP difference
            number_of_pokemons: Total number of pokemon per team
            starting_value: Base reward value
            status_value: Weight for status condition difference
            victory_value: Bonus for winning the battle
        """
        if battle.finished:
            return victory_value if battle.won else -victory_value

        # Initialize reward with starting value
        reward = starting_value

        # 1. Fainted pokemon component
        fainted_team = len([mon for mon in battle.team.values() if mon.fainted])
        fainted_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted])

        # Normalize by number of pokemon and apply weight
        fainted_diff = (fainted_opponent - fainted_team) / number_of_pokemons
        reward += fainted_value * fainted_diff

        # 2. HP component
        team_hp_fraction = 0.0
        team_alive = 0
        for mon in battle.team.values():
            if not mon.fainted and mon.current_hp_fraction is not None:
                team_hp_fraction += mon.current_hp_fraction
                team_alive += 1

        opponent_hp_fraction = 0.0
        opponent_alive = 0
        for mon in battle.opponent_team.values():
            if not mon.fainted and mon.current_hp_fraction is not None:
                opponent_hp_fraction += mon.current_hp_fraction
                opponent_alive += 1

        # Normalize HP by number of pokemon and apply weight
        if team_alive > 0 and opponent_alive > 0:
            team_avg_hp = team_hp_fraction / number_of_pokemons
            opponent_avg_hp = opponent_hp_fraction / number_of_pokemons
            hp_diff = opponent_avg_hp - team_avg_hp  # Positive if opponent has more HP
            reward += hp_value * hp_diff

        # 3. Status conditions component
        team_status_count = 0
        opponent_status_count = 0

        for mon in battle.team.values():
            if not mon.fainted and mon.status is not None:
                team_status_count += 1

        for mon in battle.opponent_team.values():
            if not mon.fainted and mon.status is not None:
                opponent_status_count += 1

        # Normalize status count and apply weight (negative status is bad for us)
        status_diff = (team_status_count - opponent_status_count) / number_of_pokemons
        reward -= status_value * status_diff  # Subtract because status is negative

        return reward

    def compute_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

    def choose_move(self, battle: AbstractBattle):
        current_state = self.embed_battle(battle)

        # Store experience from previous step
        if self.prev_state is not None and self.prev_action is not None and self.prev_battle_id == battle.battle_tag:
            reward = self.compute_reward(battle)
            done = battle.finished

            self.buffer.push(self.prev_state, self.prev_action, reward, current_state, done)

        # Choose action
        obs = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        if random.random() < self.epsilon:
            action = self._get_random_action(battle)
        else:
            with torch.no_grad():
                q = self.q_net(obs.unsqueeze(0))
            action = q.argmax(dim=1).item()

        # Store current state and action for next step
        self.prev_state = current_state
        self.prev_action = action
        self.prev_battle_id = battle.battle_tag

        return self._action_to_move(action, battle)

    def choose_switch(self, battle: AbstractBattle):
        return self.choose_move(battle)

    def _get_random_action(self, battle: AbstractBattle):
        """Get a random valid action index"""
        total_options = len(battle.available_moves) + len(battle.available_switches)
        return random.randint(0, total_options)

    def _action_to_move(self, action, battle: AbstractBattle):
        options = battle.available_moves + battle.available_switches
        if len(options) == 0:
            # Fallback to random move if no options available
            return self.choose_random_move(battle)
        move = options[action % len(options)]
        return self.create_order(move)

    async def _battle_finished_callback(self, battle: AbstractBattle):
        """Called when a battle finishes - store final experience"""
        if self.prev_state is not None and self.prev_action is not None and self.prev_battle_id == battle.battle_tag:
            final_reward = self.compute_reward(battle)
            final_state = self.embed_battle(battle)

            self.buffer.push(
                self.prev_state,
                self.prev_action,
                final_reward,
                final_state,
                True,  # Battle is finished
            )

        # Reset for next battle
        self.prev_state = None
        self.prev_action = None
        self.prev_battle_id = None

        await super()._battle_finished_callback(battle)


def train_epoch(
    agent,
    target_net,
    buffer,
    optimizer,
    batch_size: int = 4,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 1e-4,
    update_target_every: int = 1000,
):
    mlflow.log_params({"gamma": 0.99, "batch_size": 4, "lr": 1e-3})
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


def train_step(agent, target_net, buffer, optimizer, batch_size=4, gamma=0.99):
    if len(buffer) < batch_size:
        return

    s, a, r, s2, done = buffer.sample(batch_size)

    # Déplace tous les tensors du batch
    s = s.to(device)
    a = a.to(device)
    r = r.to(device)
    s2 = s2.to(device)
    done = done.to(device)

    q_values = agent.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q = target_net(s2).max(1)[0]
        targets = r + gamma * (1 - done) * max_next_q

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


async def train_agent():
    for epoch in range(10):  # 10 epochs
        logger.info(f"Epoch {epoch + 1} started.")
        epoch_losses = []

        for battle_num in range(100):  # 100 battles per epoch
            logger.info(f"  Battle {battle_num + 1} started.")
            await agent.battle_against(opponent, n_battles=1)
            # Perform training after each battle
            loss = train_step(agent, target_net, buffer, optimizer)
            if loss is not None:
                epoch_losses.append(loss)

            logger.info(f"  Battle {battle_num + 1} finished.")

        mlflow.log_metrics(
            {
                "lose_rate": agent.n_lost_battles / agent.n_finished_battles,
                "win_rate": agent.n_won_battles / agent.n_finished_battles,
                "avg_loss": np.mean(epoch_losses) if epoch_losses else 0,
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
    q_net = QNetwork(input_dim, output_dim)
    q_net = q_net.cuda()
    target_net = QNetwork(input_dim, output_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net = target_net.cuda()
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    account_configuration_agent = AccountConfiguration(
        username=f"SimpleRL{np.random.randint(1e5)}",
        password=None,
    )
    agent = SimpleRLAgent(
        account_configuration=account_configuration_agent,
        battle_format="gen9randombattle",
        q_net=q_net,
        buffer=buffer,
        device=device,
    )
    account_configuration_oppo = AccountConfiguration(
        username=f"Random{np.random.randint(1e5)}",
        password=None,
    )
    opponent = RandomPlayer(account_configuration=account_configuration_oppo, battle_format="gen9randombattle")

    asyncio.run(train_agent())

    logger.debug(f"Player {agent.username} won {agent.n_won_battles} out of {agent.n_finished_battles} played")

    mlflow.log_params({"opponent": type(opponent).__name__})

    mlflow.pytorch.log_model(
        pytorch_model=agent.q_net, name="dqn_model", input_example=np.zeros((1, 1, 10), dtype=np.float32)
    )
    save_model(agent.q_net, "pokemon_torch_rl")
