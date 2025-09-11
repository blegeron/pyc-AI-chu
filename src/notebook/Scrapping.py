import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys, ActionChains
import time
import os


MORNINGSTAR_URL = "https://www.morningstar.com/"

service = webdriver.FirefoxService(service_args=['--profile-root', '/home/user_jodert/snap/firefox/common/.mozilla/firefox/19m7b3kx.jodert'])

def get_html_selenium_firefox(url):
    driver = webdriver.Firefox(service=service)
    driver.get(url)
    company = input("Which company ?")
    input_div = driver.find_element(By.TAG_NAME, 'input')
    input_div.send_keys(company)
    ActionChains(driver)\
        .pause(1)\
        .send_keys(Keys.ENTER)\
        .perform()
    time.sleep(2)

    # Use one of the above options, e.g.:
    key_metrics = driver.find_element(By.XPATH, "//span[contains(text(), 'Key Metrics')]")
    driver.execute_script("arguments[0].click();", key_metrics.find_element(By.XPATH, "./ancestor::a"))

    time.sleep(2)
    html = driver.page_source
    driver.quit()
    return html


def get_soup(html):
    return BeautifulSoup(html)

def main():
    html = get_html_selenium_firefox(MORNINGSTAR_URL)    

    # Define the directory and filename
    directory = "/home/user_jodert/code/joderthibaud/Personal/pyc-AI-chuvTHJ/data"
    filename = "morningstar_page.html"  # You can customize the filename

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Write the soup to a file
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as file:
        file.write(get_soup(html).prettify())

    print(f"HTML saved to {os.path.join(directory, filename)}")

main()