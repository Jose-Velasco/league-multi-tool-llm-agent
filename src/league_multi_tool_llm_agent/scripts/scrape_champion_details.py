# %%
import json
from typing import Any

import requests
import scrapy
from bs4 import BeautifulSoup
from scrapy.http import Response

from league_multi_tool_llm_agent.models.champion import (
    Champion,
    ChampionAbilities,
    ChampionAbility,
    ChampionDifficulty,
    ChampionRole,
    ChampionSkin,
)


# %%
def get_next_data(url, headers):
    res = requests.get(url, headers=headers)
    html = res.text

    marker = '<script id="__NEXT_DATA__" type="application/json">'
    start = html.find(marker) + len(marker)
    end = html.find("</script>", start)

    return json.loads(html[start:end])


def extract_opgg_ai_summary(soup: BeautifulSoup) -> str | None:
    """
    Extract the "AI tips summary" text block from an OP.GG champion page.

    Args:
        soup (BeautifulSoup): Parsed HTML of the page.

    Returns:
        str | None: The extracted summary text if found, otherwise None.
    """
    strong = soup.find("strong", string="AI tips summary")  # pyright: ignore[reportCallIssue, reportArgumentType]
    if not strong:
        return None

    parent = strong.find_parent()

    while parent:
        candidate = parent.find_next("div", class_=lambda c: c and "break-words" in c)
        if candidate:
            return candidate.get_text(" ", strip=True)
        parent = parent.find_parent()

    return None


def extract_selected_role(soup: BeautifulSoup) -> str | None:
    """
    Extract the currently selected role (e.g., 'mid', 'top') from the OP.GG role selector.

    By default it selects the champions role in op.gg "champions/{champion}/build" path first far left one.
    This is the role that the champions that has the highest percentage of been seen in.

    Args:
        soup (BeautifulSoup): Parsed HTML of the page.

    Returns:
        str | None: The selected role value if found, otherwise None.
    """
    select = soup.find("select", id="selectPosition")
    if not select:
        return None

    selected_option = select.find("option", selected=True)
    if selected_option:
        return selected_option.get("value")  # pyright: ignore[reportReturnType]

    first_option = select.find("option")
    return first_option.get("value") if first_option else None  # pyright: ignore[reportReturnType]


# %%
# hp1_2_url =  "https://www.leagueoflegends.com/en-us/champions/"
champion_details_url = "https://www.leagueoflegends.com/en-us/champions/ahri/"
headers = {
    "User-Agent": "MyBot/1.0 (contact@example.com)"  # Identify your bot and contact info
}

champ_detail_payload = get_next_data(champion_details_url, headers)
champ_data = champ_detail_payload["props"]["pageProps"]["page"]


# %%
opgg_url = "https://op.gg/lol/champions/ahri/build/mid"
opgg_payload = requests.get(opgg_url, headers=headers)
print(opgg_payload.status_code)

soup = BeautifulSoup(opgg_payload.text, "lxml")

# %%
# summary = extract_opgg_ai_summary(soup)


# if summary:
#     print(summary)
# else:
#     print("AI tips summary not found")

# role = extract_selected_role(soup)

if role:
    print(role)  # "mid"
else:
    print("Role not found")

"""champ abilities"""
# champ_data["blades"][2]
"""champ skins"""
# champ_data["blades"][3]["groups"][0]["label"]
# champ_data["blades"][3]["groups"][0]["content"]["media"]["url"]
# skins
# %%

profile_card_image_url = "https://www.leagueoflegends.com/en-us/champions/"
official_lol_profile_details_website_url = (
    "https://www.leagueoflegends.com/en-us/champions/"
)

name = champ_data["blades"][1]["title"]
subtitle = champ_data["blades"][1]["subtitle"]
play_styles = [role["name"] for role in champ_data["blades"][1]["role"]["roles"]]

difficulty = champ_data["blades"][1]["difficulty"]["value"]
difficulty_name = champ_data["blades"][1]["difficulty"]["name"]

description = champ_data["blades"][1]["description"]["body"]

op_gg_link = next(
    (
        link["action"]["payload"]["url"]
        for link in champ_data["blades"][1]["links"]
        if link["title"] == "Op.gg"
    ),
    None,
)

skins = [
    (skin["label"], skin["content"]["media"]["url"])
    for skin in champ_data["blades"][3]["groups"]
]

op_gg_summary = extract_opgg_ai_summary(soup)
if op_gg_summary is None:
    op_gg_summary = "Not Found!"

main_role = extract_selected_role(soup)
if main_role is None:
    main_role = "unknown"

main_role = ChampionRole(main_role)
# %%

# Champion(
#     name=name,
#     profile_card_image_url=profile_card_image_url,  # pyright: ignore[reportArgumentType]
#     official_lol_profile_details_website_url=official_lol_profile_details_website_url,  # pyright: ignore[reportArgumentType]
#     subtitle=subtitle,
#     play_styles=play_styles,
#     difficulty=ChampionDifficulty(
#         difficulty=difficulty, difficulty_name=difficulty_name
#     ),
#     description=description,
#     op_gg_link=op_gg_link,  # pyright: ignore[reportArgumentType]
#     op_gg_summary=op_gg_summary,
#     skins=[ChampionSkin(name=name, img_url=img_url) for name, img_url in skins],
#     main_role=main_role,
# )


# %%
class ChampionSpider(scrapy.Spider):
    name = "champion"
    start_urls = ["https://www.leagueoflegends.com/en-us/champions/"]

    custom_settings = {
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/135.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            # "Referer": "https://op.gg/",
        },
        "ITEM_PIPELINES": {
            "league_multi_tool_llm_agent.utils.scrape.ChampionImagePipeline": 300,
        },
        # default but can be changed
        "IMAGES_STORE": ".data/images",
        # defult
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.5,
    }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)

        images_store = kwargs.get("images_store")
        autothrottle = kwargs.get("autothrottle_enabled")

        # "data/images"
        if images_store:
            crawler.settings.set("IMAGES_STORE", images_store, priority="spider")

        # True
        if autothrottle is not None:
            crawler.settings.set(
                "AUTOTHROTTLE_ENABLED", autothrottle, priority="spider"
            )

        return spider

    def parse(self, response: Response):
        page_json_payload = json.loads(
            response.css("script#__NEXT_DATA__ ::text").get("")
        )

        champion_card_metadata = page_json_payload["props"]["pageProps"]["page"][
            "blades"
        ][2]["items"]

        champions: dict[str, dict[str, Any]] = {}
        for champion_json in champion_card_metadata:
            champion = {
                "name": champion_json["title"],
                "profile_card_image_url": champion_json["media"]["url"],
                "official_lol_profile_details_website_url": f"https://www.leagueoflegends.com{champion_json['action']['payload']['url']}",
            }
            champions[champion["name"]] = champion

        for champion_name, champion_data in champions.items():
            champ_details_req = scrapy.Request(
                url=champion_data["official_lol_profile_details_website_url"],
                callback=self.get_champion_details,
                cb_kwargs={"champion": champion_data},
            )
            yield champ_details_req

        # TODO:
        #       test if the images are auto downloaded via ChampionImagePipeline
        #       can use shell to test scrape
        #       uv run scrapy shell "https://www.leagueoflegends.com/en-us/champions/"
        #       uv run scrapy shell "https://www.leagueoflegends.com/en-us/champions/ahri/"
        #       opgg needs headers to work
        #       uv run scrapy shell "https://op.gg/lol/champions/mel/build"
        #       to run:
        #           scrapy crawl champion -O data/champions.jsonl
        #           scrapy crawl champion -a images_store=.data/images -O data/champions.jsonl -s LOG_LEVEL=INFO

    def get_champion_details(self, response: Response, champion: dict[str, Any]):
        champ_details_payload = json.loads(
            response.css("script#__NEXT_DATA__ ::text").get("")
        )
        champ_data = champ_details_payload["props"]["pageProps"]["page"]["blades"]
        name = champ_data[1]["title"]

        subtitle = champ_data[1]["subtitle"]
        play_styles = [role["name"] for role in champ_data[1]["role"]["roles"]]

        difficulty = champ_data[1]["difficulty"]["value"]
        difficulty_name = champ_data[1]["difficulty"]["name"]

        champion_difficulty = ChampionDifficulty(
            difficulty=difficulty, difficulty_name=difficulty_name
        )

        description = champ_data[1]["description"]["body"]
        op_gg_link = next(
            (
                link["action"]["payload"]["url"]
                for link in champ_data[1]["links"]
                if link["title"] == "Op.gg"
            ),
            "",
        )

        champ_abilities_dict = {}
        for ability_metadata in champ_data[2]["groups"]:
            ability_data = ability_metadata["content"]

            ability_name = ability_data["title"]
            ability_subtitle: str = ability_data["subtitle"]
            ability_img_icon = ability_metadata["thumbnail"]["url"]
            ability_description_high_level = ability_data["description"]["body"]

            # TODO: get this data from opgg
            ability_description_details = None
            ability = ChampionAbility(
                name=ability_name,
                subtitle=ability_subtitle,
                img_icon=ability_img_icon,
                description_high_level=ability_description_high_level,
                description_details=ability_description_details,
            )
            champ_abilities_dict[ability_subtitle.lower()] = ability

        champ_abilities = ChampionAbilities(
            passive=champ_abilities_dict["passive"],
            Q=champ_abilities_dict["q"],
            W=champ_abilities_dict["w"],
            E=champ_abilities_dict["e"],
            R=champ_abilities_dict["r"],
        )

        skins: list[ChampionSkin] = []
        for skin_data in champ_data[3]["groups"]:
            skin_name, skin_img_url = (
                skin_data["label"],
                skin_data["content"]["media"]["url"],
            )
            skin = ChampionSkin(
                champion_name=name, skin_name=skin_name, img_url=skin_img_url
            )
            skins.append(skin)

        champion["subtitle"] = subtitle
        champion["play_styles"] = play_styles
        champion["difficulty"] = champion_difficulty
        champion["description"] = description
        champion["op_gg_link"] = op_gg_link
        champion["abilities"] = champ_abilities
        champion["skins"] = skins
        champion["skins"] = skins

        yield scrapy.Request(
            url=op_gg_link,
            callback=self.parse_opgg_champ_details,
            cb_kwargs={"champion": champion},
        )

    def parse_opgg_champ_details(self, response: Response, champion: dict[str, Any]):
        # op_gg_summary = response.xpath(
        #     '//*[@id="content-header"]/div[2]/div/div[1]/div[2]/div/div[2]/text()'
        # ).get()

        op_gg_summary = response.css("#content-header .break-words::text").get()
        main_role = response.xpath('//*[@id="selectPosition"]/option/text()').getall()[
            0
        ]

        # By default it selects the champions role in op.gg "champions/{champion}/build" path first far left one.
        # This is the role that the champions that has the highest percentage of been seen in.
        main_role = ChampionRole(main_role)

        champion["op_gg_summary"] = op_gg_summary
        champion["main_role"] = main_role

        yield Champion.model_validate(champion)


"""
# use to test on one item(champions)
scrapy crawl champion \
  -O ./data/champions.jsonl \
  -s IMAGES_STORE=./data/images \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.0 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=2 \
  -s LOG_LEVEL=INFO \
  -s CLOSESPIDER_ITEMCOUNT=3

"""
"""
# use to run full spider
scrapy crawl champion \
  -O ./data/champions.jsonl \
  -s IMAGES_STORE=data/images \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.0 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=2 \
  -s LOG_LEVEL=INFO
"""
