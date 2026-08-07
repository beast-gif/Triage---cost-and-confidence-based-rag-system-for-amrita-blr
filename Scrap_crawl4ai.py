# scraper.py
import asyncio
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

async def scrape_url(crawler: AsyncWebCrawler, url: str) -> dict:
    """
    Scrapes a single URL and returns both raw HTML and metadata needed
    for the extraction step.
    """
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,   # always fetch fresh, don't use crawl4ai's own cache
        wait_for="css:body",          # wait for page to render before grabbing HTML
        page_timeout=30000,
    )

    result = await crawler.arun(url=url, config=config)

    if not result.success:
        return {
            "url": url,
            "success": False,
            "error": result.error_message,
        }

    return {
        "url": url,
        "success": True,
        "html": result.html,                 # raw HTML -> feeds extract_chunks()
        "markdown": result.markdown,          # available if you ever want it instead
        "title": result.metadata.get("title", "") if result.metadata else "",
        "status_code": result.status_code,
    }


async def scrape_all(urls: list[str], concurrency: int = 5) -> list[dict]:
    """
    Scrapes a list of URLs with bounded concurrency so you don't hammer
    the admissions site.
    """
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with AsyncWebCrawler() as crawler:

        async def bound_scrape(url):
            async with semaphore:
                return await scrape_url(crawler, url)

        tasks = [bound_scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results


def save_raw_html(results: list[dict], out_dir: str = "raw_html"):
    """
    Optional: cache raw HTML to disk so extraction/debugging doesn't
    require re-scraping every time you tweak extract_chunks().
    """
    Path(out_dir).mkdir(exist_ok=True)
    for r in results:
        if not r["success"]:
            print(f"[FAILED] {r['url']} — {r.get('error')}")
            continue
        safe_name = r["url"].replace("https://", "").replace("/", "_").strip("_")
        with open(f"{out_dir}/{safe_name}.html", "w", encoding="utf-8") as f:
            f.write(r["html"])
        with open(f"{out_dir}/{safe_name}.meta.json", "w", encoding="utf-8") as f:
            json.dump({"url": r["url"], "title": r["title"]}, f)


if __name__ == "__main__":
    with open("seed_urls.json") as f:
        urls = json.load(f)

    results = asyncio.run(scrape_all(urls))
    save_raw_html(results)

    print(f"Scraped {sum(r['success'] for r in results)}/{len(results)} pages successfully")