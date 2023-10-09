# webscraping_lib.py

import time
import random
import requests
from bs4 import BeautifulSoup
from robotexclusionrulesparser import RobotExclusionRulesParser


class WebScraper:
    USER_AGENTS = [
        # List of user-agents to rotate through
        # #todo: Add more user-agents for diversity
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    ]

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.set_user_agent()
        self.robots_parser = RobotExclusionRulesParser()
        self.load_robots_txt()

    def set_user_agent(self):
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({'User-Agent': user_agent})

    def load_robots_txt(self):
        robots_txt_url = self.base_url + "/robots.txt"
        response = self.session.get(robots_txt_url)
        self.robots_parser.parse(response.text)

    def can_fetch(self, path):
        return self.robots_parser.is_allowed(self.session.headers['User-Agent'], path)

    def get(self, url, retries=3, delay=1, max_delay=30):
        for i in range(retries):
            if not self.can_fetch(url.replace(self.base_url, "")):
                print(f"Fetching {url} disallowed by robots.txt")
                return None
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
                if i < retries - 1:  # Don't sleep after the last attempt
                    time.sleep(min(delay * (2 ** i), max_delay))
        return None

    def parse_json(self, response):
        return response.json()

    def parse_html(self, response):
        return BeautifulSoup(response.text, 'html.parser')

    def parse_xml(self, response):
        return BeautifulSoup(response.text, 'xml')

    def parse_text(self, response):
        return response.text

# Usage:
# scraper = WebScraper("https://example.com")
# response = scraper.get("https://example.com/somepage")
# if response:
#     soup = scraper.parse_html(response)
