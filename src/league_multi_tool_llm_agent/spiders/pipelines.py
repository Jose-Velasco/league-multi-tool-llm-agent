from pathlib import PurePosixPath

from scrapy import Request
from scrapy.pipelines.images import ImagesPipeline
from scrapy.pipelines.media import MediaPipeline
from scrapy.utils.httpobj import urlparse_cached

from league_multi_tool_llm_agent.models.champion import (
    Champion,
)


class ChampionImagePipeline(ImagesPipeline):
    """Download champion-related images and attach local paths back to the item."""

    def get_media_requests(self, item, info: MediaPipeline.SpiderInfo) -> list[Request]:
        """Build image download requests for champion, ability, and skin images."""
        requests: list[Request] = []

        if not isinstance(item, Champion):
            return requests

        champion_slug = self._safe_name(item.name)

        # Champion profile card image
        requests.append(
            Request(
                url=str(item.profile_card_image_url),
                meta={
                    "image_kind": "profile",
                    "champion_name": champion_slug,
                },
            )
        )

        # Ability icons
        ability_map = {
            "passive": item.abilities.passive,
            "Q": item.abilities.Q,
            "W": item.abilities.W,
            "E": item.abilities.E,
            "R": item.abilities.R,
        }

        for key, ability in ability_map.items():
            requests.append(
                Request(
                    url=str(ability.img_icon),
                    meta={
                        "image_kind": "ability",
                        "champion_name": champion_slug,
                        "ability_key": key,
                        "ability_name": self._safe_name(ability.name),
                    },
                )
            )

        for skin in item.skins:
            requests.append(
                Request(
                    url=str(skin.img_url),
                    meta={
                        "image_kind": "skin",
                        "champion_name": champion_slug,
                        "skin_name": self._safe_name(skin.skin_name),
                    },
                )
            )

        return requests

    def file_path(
        self, request: Request, response=None, info=None, *, item=None
    ) -> str:
        """Return a path relative to IMAGES_STORE for each downloaded image."""
        file_extension = PurePosixPath(urlparse_cached(request).path).suffix or ".jpg"

        champion_name = request.meta.get("champion_name", "unknown_champion")
        image_kind = request.meta.get("image_kind", "misc")

        if image_kind == "profile":
            return f"{champion_name}/profile{file_extension}"

        if image_kind == "ability":
            ability_key = request.meta.get("ability_key", "unknown")
            ability_name = request.meta.get("ability_name", "unknown")
            return f"{champion_name}/abilities/{ability_key}_{ability_name}{file_extension}"

        if image_kind == "skin":
            skin_name = request.meta.get("skin_name", "unknown_skin")
            return f"{champion_name}/skins/{skin_name}{file_extension}"

        return super().file_path(request, response, info, item=item)

    def item_completed(self, results, item, info: MediaPipeline.SpiderInfo):
        """Attach downloaded local image paths back onto the Champion item."""
        if not isinstance(item, Champion):
            return item

        downloaded_by_url = {
            result["url"]: result["path"]  # pyright: ignore[reportIndexIssue]
            for success, result in results
            if success and "url" in result and "path" in result  # pyright: ignore[reportOperatorIssue]
        }

        # Attach champion profile local path
        profile_url = str(item.profile_card_image_url)
        if profile_url in downloaded_by_url:
            item.profile_card_image_path = downloaded_by_url[profile_url]

        # Attach ability icon local paths
        for ability in [
            item.abilities.passive,
            item.abilities.Q,
            item.abilities.W,
            item.abilities.E,
            item.abilities.R,
        ]:
            ability_url = str(ability.img_icon)
            if ability_url in downloaded_by_url:
                ability.img_icon_path = downloaded_by_url[ability_url]

        # Attach skin local paths
        for skin in item.skins:
            skin_url = str(skin.img_url)
            if skin_url in downloaded_by_url:
                skin.img_path = downloaded_by_url[skin_url]

        return item

    @staticmethod
    def _safe_name(value: str) -> str:
        """Convert a name into a filesystem-safe slug."""
        return (
            value.strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("'", "")
            .replace('"', "")
        )


class ToJsonPipeline:
    """
    Add a final pipeline that converts to dict.

    final conversion pipeline to not change the spiders using Pydantic models

    usage set priority in ITEM_PIPELINES of ToJsonPipeline lower than you custom ImagePipeline
    so it can save to json properly. EX: ToJsonPipeline: 900

    This helps workaround when Pydantic model fields such as HttpUrl
    can not be serialized by Scrapy's exporter. In this case we user Pydantics' model_dump to json
    to serialize for us. Add more as needed.


    Spider

    ↓

    ChampionImagePipeline (needs Pydantic)

    ↓

    ToJsonPipeline (convert to dict)

    ↓

    FeedExporter (JSONL)
    """

    def process_item(self, item):
        if isinstance(item, Champion):
            return item.model_dump(mode="json")
        return item
