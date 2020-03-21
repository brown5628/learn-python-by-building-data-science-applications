# %%
import requests as rq
# import json
from bs4 import BeautifulSoup

# %%
base_url = "https://en.wikipedia.org/wiki/List_of_World_War_II_battles"

# %%
response = rq.get(base_url)
response.content

# %%
soup = BeautifulSoup(response.content, "html.parser")
soup

# %%


def get_dom(url):
    response = rq.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")
