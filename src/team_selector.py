import os
import random
from datetime import datetime

from loguru import logger

from utils.registry import pokemon_data, save_team


def generate_team_str(poke1=None, poke2=None, poke3=None, poke4=None, poke5=None, poke6=None):
    # Load the pokemons_selector.csv file with explicit encoding
    pokemons_df = pokemon_data()

    # List of input pokemons
    input_pokemons = [poke1, poke2, poke3, poke4, poke5, poke6]

    # If any input is None, select random unique pokemons
    if any(poke is None for poke in input_pokemons):
        available_pokemons = pokemons_df["poke_name"].unique().tolist()
        selected_pokemons = random.sample(available_pokemons, 6)
        for i in range(6):
            if input_pokemons[i] is None:
                input_pokemons[i] = selected_pokemons.pop()

    # Ensure all pokemons are unique by replacing duplicates with random unique ones
    unique_pokemons = []
    for poke in input_pokemons:
        while poke in unique_pokemons:
            poke = random.choice(available_pokemons)
        unique_pokemons.append(poke)
    input_pokemons = unique_pokemons

    logger.info(f"Generating team with pokemons: {input_pokemons}")

    smogon_team = ""

    for poke_name in input_pokemons:
        # Find the row for the current poke_name
        poke_row = pokemons_df[pokemons_df["poke_name"] == poke_name].iloc[0]
        row_len = poke_row["row_len"]

        # Write the rows to the output file
        for i in range(1, row_len + 1):
            smogon_team += f"{poke_row[f'row{i}']}\n"

        # Add a blank line after each Pokémon
        smogon_team += "\n"

    return smogon_team


def generate_team(format: str = "gen9ou", constraint: str | None = None) -> str:
    # This function can be expanded to generate teams based on format and constraints
    return generate_team_str()


if __name__ == "__main__":
    team = generate_team_str()
    save_team(team)
