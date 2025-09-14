import os
import random
from datetime import datetime

from loguru import logger

from utils.registry import data_path, pokemon_data


def generate_team_file(poke1=None, poke2=None, poke3=None, poke4=None, poke5=None, poke6=None):
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

    # Define the output file path
    output_file_path = os.path.join(data_path, "team_file.txt")
    log_file_path = os.path.join(data_path, "team_file_log.txt")

    # Open the output file in write mode
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for poke_name in input_pokemons:
            # Find the row for the current poke_name
            poke_row = pokemons_df[pokemons_df["poke_name"] == poke_name].iloc[0]
            row_len = poke_row["row_len"]

            # Write the rows to the output file
            for i in range(1, row_len + 1):
                output_file.write(f"{poke_row[f'row{i}']}\n")

            # Add a blank line after each Pokémon
            output_file.write("\n")

    # Log the execution
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - Function called with pokemons: {input_pokemons}\n"

    # Open the log file in append mode with encoding
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)

    logger.info(f"Team file generated at {output_file_path}")
    logger.info(f"Execution logged at {log_file_path}")


if __name__ == "__main__":
    generate_team_file()
