import json
from datetime import datetime
from pathlib import Path
from typing import Any

import scrapy
from scrapy.http import Response

from league_multi_tool_llm_agent.models.champion import (
    Champion,
    ChampionAbilities,
    ChampionAbility,
    ChampionDifficulty,
    ChampionRole,
    ChampionSkin,
)


class ChampionSpider(scrapy.Spider):
    name = "champion"
    start_urls = ["https://www.leagueoflegends.com/en-us/champions/"]

    custom_settings = {
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/135.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "ITEM_PIPELINES": {
            "league_multi_tool_llm_agent.spiders.pipelines.ChampionImagePipeline": 300,
            "league_multi_tool_llm_agent.spiders.pipelines.ToJsonPipeline": 900,
        },
        # default but can be changed
        "IMAGES_STORE": ".data/images",
        # defult
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.50,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        # "CONCURRENT_REQUESTS": 12,
    }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("./data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # create dirs if missing
        log_file = log_dir / f"{spider.name}_{timestamp}.log"
        # log_file = f"./data/logs/{spider.name}_{timestamp}.log"

        crawler.settings.set("LOG_FILE", log_file, priority="spider")

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

        for _champion_name, champion_data in champions.items():
            champ_details_req = scrapy.Request(
                url=champion_data["official_lol_profile_details_website_url"],
                callback=self.get_champion_details,
                cb_kwargs={"champion": champion_data},
            )
            yield champ_details_req

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

        # Riot is giving a malformed OP.GG URL or this does not parse properly for all champions
        # op_gg_link = next(
        #     (
        #         link["action"]["payload"]["url"]
        #         for link in champ_data[1]["links"]
        #         if link["title"] == "Op.gg"
        #     ),
        #     "",
        # )
        # champ_slug = self.to_opgg_slug(name)
        # op_gg_link = f"https://op.gg/lol/champions/{champ_slug}/build"

        raw_opgg_link = next(
            (
                link["action"]["payload"]["url"]
                for link in champ_data[1]["links"]
                if link["title"] == "Op.gg"
            ),
            "",
        )

        op_gg_link = self.normalize_opgg_link(raw_opgg_link, name)

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
        main_role = ChampionRole(main_role.lower())

        champion["op_gg_summary"] = op_gg_summary
        champion["main_role"] = main_role

        yield Champion.model_validate(champion)

    def normalize_opgg_link(self, raw_url: str, champion_name: str) -> str:
        """Return a valid OP.GG build URL, repairing malformed links when needed."""
        champ_slug = champion_name.strip().lower().replace(" ", "-")
        expected = f"https://op.gg/lol/champions/{champ_slug}/build"

        if not raw_url:
            return expected

        if raw_url.endswith(f"{champ_slug}build"):
            return expected

        if "/build" not in raw_url:
            return expected

        return raw_url

    def to_opgg_slug(self, name: str) -> str:
        """Convert a champion name into an OP.GG URL slug."""
        return name.strip().lower().replace(" ", "-")


"""
# use to test on one item(champions)
uv run scrapy crawl champion \
  -s IMAGES_STORE=./data/images \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.0 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=2 \
  -s LOG_LEVEL=INFO \
  -s CLOSESPIDER_ITEMCOUNT=1
"""

"""
# use to run full spider
uv run scrapy crawl champion \
  -s IMAGES_STORE=./data/images \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.0 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=2 \
  -s LOG_LEVEL=INFO


# use to run full spider (FASTER???)
uv run scrapy crawl champion \
  -s IMAGES_STORE=./data/images \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.5 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=3 \
  -s AUTOTHROTTLE_START_DELAY=0.5 \
  -s DOWNLOAD_DELAY=0.25 \
  -s CONCURRENT_REQUESTS=12 \
  -s LOG_LEVEL=INFO
"""

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
