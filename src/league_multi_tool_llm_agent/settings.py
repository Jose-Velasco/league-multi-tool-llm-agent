"""
Scrapy settings for league_multi_tool_llm_agent.

These defaults are tuned for a small, polite scraping workflow with:
- src/ package layout
- custom image pipeline
- JSON/JSONL export support
- OP.GG-friendly request behavior
"""

BOT_NAME = "league_multi_tool_llm_agent"

SPIDER_MODULES = ["league_multi_tool_llm_agent.spiders"]
NEWSPIDER_MODULE = "league_multi_tool_llm_agent.spiders"

ROBOTSTXT_OBEY = False

CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_DELAY = 0.5
RANDOMIZE_DOWNLOAD_DELAY = True

COOKIES_ENABLED = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

RETRY_ENABLED = True
RETRY_TIMES = 2

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)

DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://op.gg/",
}

IMAGES_STORE = "./data/images"

# Optional feed export defaults.
FEEDS = {
    "./data/%(name)s_%(time)s.jsonl": {
        "format": "jsonlines",
        # "overwrite": True,
        "encoding": "utf8",
        # "indent": None,
    },
}

# Logging
LOG_LEVEL = "INFO"

# Encoding / request behavior
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
