# %%
import requests as rq
import json
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


# %%
soup = get_dom(base_url)
content = soup.select("div#mw-content-text>div.mw-parser-output", limit=1)[0]

# %%
content

# %%
fronts = content.select("div.mw-parser-output>h2")[:-1]

# %%
for front in fronts:
    print(front.text[:-6])

# %%


def _abs_link(link, base="https://en.wikipedia.org"):
    return base + link


# %%


def dictify(ul, level=0):
    result = dict()

    for li in ul.find_all("li", recursive=False):
        text = li.stripped_strings
        key = next(text)

        try:
            time = next(text).replace(":", "").strip()
        except StopIteration:
            time = None

        ul, link = li.find("ul"), li.find("a")
        if link:
            link = _abs_link(link.get("href"))

        r = {"url": link, "time": time, "level": level}

        if ul:
            r["children"] = dictify(ul, level=(level + 1))

        result[key] = r
    return result


# %%
z = dictify(fronts[0].find_next_siblings("div", "div-col columns column-width")[0].ul)
z

# %%
theaters = {}

for front in fronts:
    list_element = front.find_next_siblings("div", "div-col columns column-width")[0].ul
    theaters[front.text[:-6]] = dictify(list_element)

# %%
theaters.keys()

# %%
with open("all_battles.json", "w") as f:
    json.dump(theaters, f)

# %%
url = "https://en.wikipedia.org//wiki/Operation_Skorpion"

info = get_dom(url)

# %%
table = info.select("table.infobox.vevent")[0]

# %%


def _table_to_dict(table):
    result = {}
    for row in table.find_all("tr"):
        result[row.th.text] = row.td.get_text().strip()

    return result


# %%


def _get_main_info(table):
    main = [
        el
        for el in table.tbody.find_all("tr", recursive=False)
        if "Location" in el.get_text()
    ][0]
    return {"main": _table_to_dict(main)}


# %%
_get_main_info(table)

# %%


def _parse_row(row, names=("allies", "axis", "third party")):
    """
    parse secondary info row as dict of
    info points
    """
    cells = row.find_all("td", recursive=False)
    if len(cells) == 1:
        return {"total": cells[0].get_text(separator=" ").strip()}

    return {
        name: cell.get_text(separator=" ").strip() for name, cell in zip(names, cells)
    }


def _find_row_by_header(table, string):
    header = table.tbody.find("tr", text=string)
    if header is not None:
        return header.next_sibling


def _additional(table):
    """
    collects additional info using header
    keywords and returning data from the row below each
    """

    keywords = (
        "Belligerents",
        "Commanders and leaders",
        "Strength",
        "Casualties and losses",
    )

    result = {}
    for keyword in keywords:
        try:
            data = _find_row_by_header(table, keyword)
            if data:
                result[keyword] = _parse_row(data)
        except Exception as e:
            raise Exception(keyword, e)

    return result


# %%
_additional(table)

# %%
