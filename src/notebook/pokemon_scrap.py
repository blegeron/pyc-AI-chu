import csv
import os
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains

POKEMON_URL = "https://www.smogon.com/dex/sv/formats/ou/"

service = webdriver.FirefoxService(service_args=['--profile-root', '/home/user_jodert/snap/firefox/common/.mozilla/firefox/19m7b3kx.jodert'])


class Pokemon:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.info = {}  # Placeholder for storing fetched info

    def fetch_info(self):
        # Fetch and parse the Pokémon's dedicated page
        response = requests.get(self.url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Add logic here to extract the info you need from the soup
            # Example: Extract abilities
            # abilities = soup.find('div', class_='abilities')
            # if abilities:
            #     self.info['abilities'] = abilities.text.strip()
            # Add more logic to extract other info
            pass
        else:
            print(f"Failed to fetch info for {self.name}")

    def to_dict(self):
        # Return a dictionary of attributes to save in CSV
        return {
            'name': self.name,
            'url': self.url
            # **self.info  # Include all info attributes
        }

def get_html_selenium_firefox(url):
    driver = webdriver.Firefox(service=service)
    driver.get(url)
    time.sleep(2)  # Wait for the page to load
    html = driver.page_source
    driver.quit()
    return html

def get_soup(html):
    return BeautifulSoup(html, features="html.parser")

def extract_pokemon_names(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    pokemon_divs = soup.find_all('div', class_='PokemonAltRow-name')
    print(f"Found {len(pokemon_divs)} Pokémon divs")  # Debug print
    pokemons = []
    for div in pokemon_divs:
        a_tag = div.find('a')
        if a_tag and a_tag.has_attr('href'):
            href = a_tag['href']
            name = href.split('/')[-2]  # Extract the last part of the path
            url = f"https://www.smogon.com{href}"
            pokemons.append(Pokemon(name, url))
    return pokemons

def save_pokemons_to_csv(pokemons, directory, filename='pokemons.csv'):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Define the CSV file path
    csv_path = os.path.join(directory, filename)

    # Define the fieldnames (columns) for the CSV
    fieldnames = ['name', 'url'] + list(pokemons[0].info.keys()) if pokemons else ['name', 'url']

    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pokemon in pokemons:
            writer.writerow(pokemon.to_dict())

    print(f"Pokemons saved to {csv_path}")

def main():

    # Load data
    html = get_html_selenium_firefox(POKEMON_URL) 
    soup = get_soup(html)   
    html_content = get_soup(html).prettify()


    # Extract Pokémon names and create Pokemon objects
    pokemons = extract_pokemon_names(html_content)
    print(f"Number of Pokémon to save: {len(pokemons)}")  # Debug print

    # Fetch info for each Pokémon
    for pokemon in pokemons:
        pokemon.fetch_info()

    # Save the list of Pokémon to CSV
    save_pokemons_to_csv(
        pokemons,
        directory="/home/user_jodert/code/joderthibaud/Personal/pyc-AI-chuvTHJ/data"
    )

if __name__ == "__main__":
    main()
