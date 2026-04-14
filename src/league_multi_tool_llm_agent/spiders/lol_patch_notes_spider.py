import re
from datetime import datetime
from pathlib import Path
from typing import Any

import scrapy
from scrapy.http import Response

from league_multi_tool_llm_agent.models.patch_note import PatchNotes


class LOLPatchNotesSpider(scrapy.Spider):
    name = "lol_patch_notes"
    start_urls = ["https://www.leagueoflegends.com/en-us/news/tags/patch-notes/"]

    custom_settings = {
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/135.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        # might not work because I am not setting FILES_STORE or IMAGES_STORE (This spider implementation does not download files or imgs)
        # "ITEM_PIPELINES": {
        #     "league_multi_tool_llm_agent.spiders.pipelines.ToJsonPipeline": 900,
        # },
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 0.50,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "LOG_LEVEL": "INFO",
    }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)

        timestamp = datetime.now().strftime("%Y-%m-%d_T%H_%M_%S")
        log_dir = Path("./data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # create dirs if missing
        log_file = log_dir / f"{spider.name}_{timestamp}.log"

        crawler.settings.set("LOG_FILE", log_file, priority="spider")

        autothrottle = kwargs.get("autothrottle_enabled")
        if autothrottle is not None:
            crawler.settings.set(
                "AUTOTHROTTLE_ENABLED",
                autothrottle,
                priority="spider",
            )

        max_patch_notes = kwargs.get("max_patch_notes")
        spider.max_patch_notes = int(max_patch_notes) if max_patch_notes else 5  # pyright: ignore[reportAttributeAccessIssue]

        return spider

    def parse(self, response: Response):
        """
        Parse the patch notes listing page and follow the most recent patch note links.
        """
        patch_links = self._extract_patch_note_links(response)

        if not patch_links:
            self.logger.warning("No patch note links found on listing page.")
            return

        limited_links = patch_links[: self.max_patch_notes]  # pyright: ignore[reportAttributeAccessIssue]
        self.logger.info(
            "Following %s most recent patch note pages.", len(limited_links)
        )

        for link in limited_links:
            yield response.follow(
                url=link,
                callback=self.parse_patch_note_details,
            )

    def parse_patch_note_details(self, response: Response):
        """
        Parse a Riot patch note details page into a structured document.
        """
        slug = self.extract_patch_slug(response.url)
        patch_notes_data: dict[str, Any] = {
            "url": response.url,
            "title": response.css("title::text").get(),
            "patch_slug": slug,
            "patch_version": self.extract_patch_version(slug),
            "tagline": self._extract_tagline(response),
            "authors": self._extract_authors(response),
            "date": response.css("time::text").get(),
            "author_context": self._extract_author_context(response),
            "text_contents": self._extract_patch_note_by_section(response),
        }

        #  using pydantic for validation and it serializer because Scrapy cant handle HttpUrl serilization
        yield PatchNotes.model_validate(patch_notes_data).model_dump(mode="json")

    def extract_patch_slug(self, url: str) -> str:
        """Extract the trailing slug from a patch note URL."""
        return url.rstrip("/").split("/")[-1]

    def extract_patch_version(self, slug: str) -> str | None:
        """Extract patch version like '26.7' from a patch slug."""
        match = re.search(r"patch-(\d+)-(\d+)", slug)
        if not match:
            return None
        return f"{match.group(1)}.{match.group(2)}"

    def _extract_patch_note_links(self, response: Response) -> list[str]:
        """Extract Riot LoL patch note detail links in page order."""
        hrefs = response.css('a[href*="/news/game-updates/"]::attr(href)').getall()

        links: list[str] = []
        seen: set[str] = set()

        for href in hrefs:
            full_url = response.urljoin(href)

            # Skip TFT and generic landing pages
            if "teamfighttactics.leagueoflegends.com" in full_url:
                continue
            if full_url.rstrip("/").endswith("/news/game-updates"):
                continue

            slug = full_url.rstrip("/").split("/")[-1]
            if "patch" not in slug or "notes" not in slug:
                continue

            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)

        return links

    def _extract_tagline(self, response: Response) -> str | None:
        """Extract the patch note tagline/header summary."""
        tagline = response.xpath(
            '//*[@id="__next"]/div/main/div/div/section[2]/div[2]/div/div[1]/div/div/div/text()'
        ).get()

        if tagline:
            return tagline.strip()

        return None

    def _extract_authors(self, response: Response) -> list[str]:
        """Extract author names from the patch note page."""
        authors = response.css("div.authors ::text").getall()
        return [author.strip() for author in authors if author.strip()]

    def _extract_author_context(self, response: Response) -> str | None:
        """Extract and normalize the short editor/context note near the patch notes body."""
        parts = response.css("#patch-notes-container > blockquote ::text").getall()
        if not parts:
            return None

        text = self._clean_text(" ".join(parts))
        return text or None

    def _extract_patch_note_by_section(
        self, response: Response
    ) -> list[dict[str, str]]:
        """
        Extract patch notes grouped by section header.
        """
        patch_note_by_section: list[dict[str, str]] = []
        container_children = response.css("#patch-notes-container > *")

        it = iter(container_children)
        curr_tag = next(it, None)

        while curr_tag is not None:
            if curr_tag.root.tag == "header":
                section_title = curr_tag.css("h2::text").get()
                curr_tag = next(it, None)
                section_all_text = ""

                while curr_tag is not None and curr_tag.css("div.content-border"):
                    section_text = "".join(curr_tag.css("div ::text").getall())
                    section_text = self._clean_text(section_text)
                    if section_text:
                        section_all_text += section_text + "\n"
                    curr_tag = next(it, None)

                if section_title and section_all_text.strip():
                    patch_note_by_section.append(
                        {
                            "section_title": section_title.strip(),
                            "section_text_contents": section_all_text.strip(),
                            "full_text": f"{section_title.strip()}: {section_all_text.strip()}",
                            "section_index": str(len(patch_note_by_section)),
                        }
                    )
                continue

            curr_tag = next(it, None)

        return patch_note_by_section

    def _clean_text(self, text: str) -> str:
        """Normalize scraped text by removing excessive whitespace."""
        text = re.sub(r"^\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\t{2,}", "", text)
        text = re.sub(r"[ \xa0]{2,}", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()


# uv run scrapy shell "https://www.leagueoflegends.com/en-us/news/tags/patch-notes/"
# uv run scrapy shell "https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-7-notes/"
"""
# use to test on one item(champions)
uv run scrapy crawl lol_patch_notes \
  -a max_patch_notes=2 \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.0 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=2 \
  -s AUTOTHROTTLE_START_DELAY=1.0 \
  -s DOWNLOAD_DELAY=0.50 \
  -s CONCURRENT_REQUESTS=2 \
  -s CLOSESPIDER_ITEMCOUNT=2 \
  -s LOG_LEVEL=INFO
"""


"""
simple run command
uv run scrapy crawl lol_patch_notes -a max_patch_notes=3

# use to run full spider (FASTER???)
uv run scrapy crawl lol_patch_notes \
  -a max_patch_notes=12 \
  -s AUTOTHROTTLE_ENABLED=True \
  -s AUTOTHROTTLE_TARGET_CONCURRENCY=1.5 \
  -s CONCURRENT_REQUESTS_PER_DOMAIN=3 \
  -s AUTOTHROTTLE_START_DELAY=0.5 \
  -s DOWNLOAD_DELAY=0.25 \
  -s CONCURRENT_REQUESTS=8 \
  -s LOG_LEVEL=INFO
"""
