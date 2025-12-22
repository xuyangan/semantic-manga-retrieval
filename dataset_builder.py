"""
Dataset Builder for Manga Vectorizer

Automatically downloads manga panels to create datasets for clustering experiments.

Two modes:
1. Author-based datasets (original): Downloads by author with author/manga/chapter structure
2. Final dataset mode (new): Downloads random manga to final_dataset/manga_name/chapter structure

Configuration for final_dataset mode:
- Downloads 30 pages per manga
- Skips first 4 and last 4 pages per chapter
- Only selects manga not officially published
- Requires at least 30 valid pages total
- Skips manga already in final_dataset
"""

import requests
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field

# Import from manga_downloader
from manga_downloader import (
    BASE_URL,
    MANGA_URL,
    HEADERS,
    get_chapter_pages,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = Path("./datasets")
FINAL_DATASET_DIR = Path("./final_dataset")

# Author-based dataset config
PANELS_PER_MANGA = 20
WORKS_PER_AUTHOR = 3
MIN_CHAPTERS_PER_MANGA = 5
PAGES_TO_SKIP_START = 5
PAGES_TO_SKIP_END = 5

# Final dataset config
FINAL_PAGES_PER_MANGA = 30
FINAL_SKIP_START = 3
FINAL_SKIP_END = 3
FINAL_MIN_VALID_PAGES = 30
PAGES_PER_CHAPTER = 5  # Target ~5 pages per chapter

# Filters
OFFICIAL_LINK_KEYS = {"amazon", "ebj", "cdj", "bw", "mu"}
COLOR_TERMS = {"colored", "color", "colour", "coloured", "full color"}
EXCLUDE_TERMS = {"doujinshi", "doujin", "anthology"}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MangaWork:
    """Represents a manga work with its metadata."""
    id: str
    title: str
    author_id: str
    author_name: str
    status: str
    content_rating: str
    chapters: list = field(default_factory=list)
    valid_pages_count: int = 0


@dataclass
class Author:
    """Represents an author with their works."""
    id: str
    name: str
    works: list[MangaWork] = field(default_factory=list)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def rate_limit_sleep(seconds: float = 0.3):
    """Sleep to avoid rate limiting."""
    time.sleep(seconds)


def sanitize_filename(name: str) -> str:
    """Create a safe filename/dirname from a string."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()


def normalize_name(name: str) -> str:
    """Normalize name for comparison (lowercase, no spaces/dashes)."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def get_manga_title(manga_data: dict) -> str:
    """Extract title from manga API data, preferring English."""
    titles = manga_data["attributes"].get("title", {})
    return titles.get("en") or titles.get("ja-ro") or next(iter(titles.values()), "Unknown") if titles else "Unknown"


def is_colored_manga(title: str) -> bool:
    """Check if manga title indicates it's colored."""
    title_lower = title.lower()
    return any(term in title_lower for term in COLOR_TERMS)


def should_exclude_manga(title: str) -> bool:
    """Check if manga should be excluded based on title."""
    title_lower = title.lower()
    # Check for color terms
    if any(term in title_lower for term in COLOR_TERMS):
        return True
    # Check for excluded terms (doujinshi, etc.)
    if any(term in title_lower for term in EXCLUDE_TERMS):
        return True
    return False


def has_official_links(manga_data: dict) -> bool:
    """Check if manga has official retail links."""
    links = manga_data["attributes"].get("links", {}) or {}
    return bool(OFFICIAL_LINK_KEYS & set(links.keys()))


def get_valid_page_range(total_pages: int, skip_start: int, skip_end: int) -> list[int]:
    """Get valid page indices after skipping start/end pages."""
    valid_start = skip_start
    valid_end = total_pages - skip_end
    if valid_end <= valid_start:
        return []
    return list(range(valid_start, valid_end))


def get_existing_manga_in_final_dataset() -> set[str]:
    """Get normalized names of manga already in final_dataset."""
    existing = set()
    if FINAL_DATASET_DIR.exists():
        for item in FINAL_DATASET_DIR.iterdir():
            if item.is_dir():
                existing.add(normalize_name(item.name))
    return existing


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_valid_chapters(manga_id: str) -> list[dict]:
    """
    Get chapters for a manga, filtering out float/bonus chapters.
    Returns sorted list of chapter info dicts.
    """
    try:
        resp = requests.get(
            f"{MANGA_URL}/{manga_id}/aggregate",
            params={"translatedLanguage[]": ["en"]},
            headers=HEADERS,
            timeout=30
        )
        rate_limit_sleep()
        
        if resp.status_code != 200:
            return []
        
        data = resp.json()
    except Exception:
        return []
    
    volumes = data.get("volumes", {})
    
    if not volumes or not isinstance(volumes, dict):
        return []
    
    chapters = []
    for vol_data in volumes.values():
        vol_chapters = vol_data.get("chapters", {})
        if not vol_chapters or not isinstance(vol_chapters, dict):
            continue
        
        for ch_key, ch_data in vol_chapters.items():
            try:
                if ch_key == "none":
                    continue
                ch_num = float(ch_key)
                if ch_num != int(ch_num):  # Skip float chapters (5.5, etc.)
                    continue
                
                chapters.append({
                    "chapter": ch_key,
                    "chapter_num": int(ch_num),
                    "id": ch_data.get("id"),
                })
            except (ValueError, AttributeError):
                continue
    
    return sorted(chapters, key=lambda x: x["chapter_num"])


def fetch_author_works(author_id: str) -> tuple[str, list[dict]]:
    """Fetch author name and all their non-official manga works."""
    author_name = "Unknown"
    works = []
    
    try:
        # Get author info
        resp = requests.get(f"{BASE_URL}/author/{author_id}", headers=HEADERS, timeout=30)
        rate_limit_sleep(0.2)
        
        if resp.status_code != 200:
            return author_name, works
        
        data = resp.json()
        
        if data.get("data"):
            author_name = data["data"]["attributes"].get("name", "Unknown")
        
        # Get manga by author
        params = {
            "authors[]": [author_id],
            "limit": 100,
            "availableTranslatedLanguage[]": ["en"],
            "hasAvailableChapters": "true",
            "contentRating[]": ["safe", "suggestive"],
        }
        
        manga_resp = requests.get(MANGA_URL, params=params, headers=HEADERS, timeout=30)
        rate_limit_sleep(0.2)
        
        if manga_resp.status_code != 200:
            return author_name, works
        
        for manga in manga_resp.json().get("data", []):
            if has_official_links(manga):
                continue
            
            title = get_manga_title(manga)
            if should_exclude_manga(title):
                continue
            
            works.append({
                "id": manga["id"],
                "title": title,
                "status": manga["attributes"].get("status", "unknown"),
                "content_rating": manga["attributes"].get("contentRating", "unknown"),
            })
    except Exception as e:
        print(f"      Error fetching author {author_id}: {e}")
    
    return author_name, works


def search_manga_api(config: dict, retries: int = 2) -> list[dict]:
    """Execute a single manga search API call with retry logic."""
    params = {
        "limit": 50,
        "includes[]": ["author"],
        "availableTranslatedLanguage[]": ["en"],
        "hasAvailableChapters": "true",
        "contentRating[]": ["safe", "suggestive"],
        **config
    }
    
    for attempt in range(retries + 1):
        try:
            resp = requests.get(MANGA_URL, params=params, headers=HEADERS, timeout=30)
            rate_limit_sleep()
            
            if resp.status_code == 429:  # Rate limited
                print(f"      Rate limited, waiting...")
                time.sleep(5)
                continue
            
            if resp.status_code != 200:
                if attempt < retries:
                    time.sleep(1)
                    continue
                return []
            
            return resp.json().get("data", [])
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            print(f"      Warning: Search failed - {e}")
            return []
    
    return []


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_pages_from_chapters(
    chapters: list[dict],
    manga_dir: Path,
    num_pages: int,
    skip_start: int,
    skip_end: int,
    pages_per_chapter_range: tuple[int, int] = (2, 4)
) -> int:
    """
    Download pages from chapters, spreading across multiple chapters.
    Returns number of pages downloaded.
    """
    random.shuffle(chapters)
    downloaded_count = 0
    
    for chapter in chapters:
        if downloaded_count >= num_pages:
            break
        
        remaining = num_pages - downloaded_count
        pages_to_get = min(remaining, random.randint(*pages_per_chapter_range))
        
        rate_limit_sleep(0.3)
        chapter_data = get_chapter_pages(chapter["id"], quiet=True)
        
        if not chapter_data:
            continue
        
        pages = chapter_data["pages"]
        total = chapter_data["total"]
        base_url = chapter_data["base_url"]
        chapter_hash = chapter_data["hash"]
        
        valid_indices = get_valid_page_range(total, skip_start, skip_end)
        if not valid_indices:
            continue
        
        pages_to_get = min(pages_to_get, len(valid_indices))
        selected_indices = sorted(random.sample(valid_indices, pages_to_get))
        
        # Create chapter directory
        chapter_dir = manga_dir / f"chapter_{chapter['chapter']}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Download pages
        for page_idx in selected_indices:
            page_filename = pages[page_idx]
            image_url = f"{base_url}/data/{chapter_hash}/{page_filename}"
            
            try:
                rate_limit_sleep(0.2)
                img_resp = requests.get(image_url, headers=HEADERS)
                img_resp.raise_for_status()
                
                ext = Path(page_filename).suffix or ".jpg"
                filepath = chapter_dir / f"page_{page_idx + 1:03d}{ext}"
                
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                
                downloaded_count += 1
            except Exception:
                pass
    
    return downloaded_count


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def estimate_valid_pages(
    manga_id: str,
    skip_start: int,
    skip_end: int,
    sample_size: int = 8
) -> tuple[list[dict], int, float]:
    """
    Estimate total valid pages by sampling chapters.
    Returns (chapters, estimated_total_pages, success_rate).
    """
    chapters = get_valid_chapters(manga_id)
    
    if not chapters:
        return [], 0, 0.0
    
    sample_size = min(sample_size, len(chapters))
    sample_chapters = random.sample(chapters, sample_size)
    
    total_valid_pages = 0
    successful_samples = 0
    
    for ch in sample_chapters:
        rate_limit_sleep(0.25)
        chapter_data = get_chapter_pages(ch["id"], quiet=True)
        
        if not chapter_data:
            continue
        
        valid_pages = chapter_data["total"] - skip_start - skip_end
        if valid_pages >= 2:
            total_valid_pages += valid_pages
            successful_samples += 1
    
    if successful_samples == 0:
        return chapters, 0, 0.0
    
    success_rate = successful_samples / sample_size
    avg_valid = total_valid_pages / successful_samples
    estimated_total = int(avg_valid * len(chapters) * success_rate)
    
    return chapters, estimated_total, success_rate


def validate_manga_for_final_dataset(manga: dict) -> tuple[bool, list[dict], int]:
    """Validate manga has enough valid pages for final dataset."""
    print(f"   Validating: {manga['title'][:50]}...", end=" ")
    
    chapters, estimated_pages, success_rate = estimate_valid_pages(
        manga["id"], FINAL_SKIP_START, FINAL_SKIP_END, sample_size=10
    )
    
    if len(chapters) < 3:
        print(f"❌ Only {len(chapters)} chapters")
        return False, [], 0
    
    if estimated_pages < FINAL_MIN_VALID_PAGES:
        print(f"❌ ~{estimated_pages} pages (need {FINAL_MIN_VALID_PAGES}+)")
        return False, [], 0
    
    print(f"✓ {len(chapters)} chapters, ~{estimated_pages} valid pages")
    return True, chapters, estimated_pages


def validate_manga_for_dataset(work: MangaWork) -> bool:
    """Validate manga has enough valid pages for author-based dataset."""
    chapters, estimated_pages, success_rate = estimate_valid_pages(
        work.id, PAGES_TO_SKIP_START, PAGES_TO_SKIP_END, sample_size=8
    )
    
    if len(chapters) < MIN_CHAPTERS_PER_MANGA:
        print(f"❌ Only {len(chapters)} chapters (need {MIN_CHAPTERS_PER_MANGA}+)")
        return False
    
    if success_rate < 0.5:
        print(f"❌ Too many unavailable chapters (success: {success_rate:.0%})")
        return False
    
    required_pages = int(PANELS_PER_MANGA * 1.5)
    if estimated_pages < required_pages:
        print(f"❌ ~{estimated_pages} est. pages (need {required_pages}+ for safety)")
        return False
    
    work.chapters = chapters
    work.valid_pages_count = estimated_pages
    print(f"✓ {len(chapters)} ch, ~{estimated_pages} pages (success: {success_rate:.0%})")
    return True


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def search_random_manga(limit: int = 100, exclude_existing: bool = True) -> list[dict]:
    """Search for random non-official manga with English translations."""
    print(f"   🔍 Searching for random manga...")
    
    existing = get_existing_manga_in_final_dataset() if exclude_existing else set()
    if existing:
        print(f"      Found {len(existing)} manga already in final_dataset (will skip)")
    
    found_manga = []
    found_ids = set()
    
    # Generate truly random search configs
    search_configs = []
    
    orderings = [
        {"order[followedCount]": "desc"},
        {"order[rating]": "desc"},
        {"order[createdAt]": "desc"},
        {"order[updatedAt]": "desc"},
        {"order[latestUploadedChapter]": "desc"},
    ]
    
    # Random offsets across different orderings
    for _ in range(20):
        ordering = random.choice(orderings)
        offset = random.randint(0, 5000)
        search_configs.append({**ordering, "offset": offset})
    
    random.shuffle(search_configs)
    
    for config in search_configs:
        if len(found_manga) >= limit:
            break
        
        for manga in search_manga_api(config):
            if manga["id"] in found_ids:
                continue
            if has_official_links(manga):
                continue
            
            title = get_manga_title(manga)
            
            # Skip colored manga, doujinshi, anthologies
            if should_exclude_manga(title):
                continue
            if normalize_name(title) in existing:
                continue
            
            # Get author name
            author_name = "Unknown"
            for rel in manga.get("relationships", []):
                if rel["type"] == "author" and "attributes" in rel:
                    author_name = rel["attributes"].get("name", "Unknown")
                    break
            
            found_ids.add(manga["id"])
            found_manga.append({
                "id": manga["id"],
                "title": title,
                "author": author_name,
                "status": manga["attributes"].get("status", "unknown"),
                "content_rating": manga["attributes"].get("contentRating", "unknown"),
            })
            
            if len(found_manga) >= limit:
                break
    
    random.shuffle(found_manga)
    print(f"      ✓ Found {len(found_manga)} candidate manga")
    return found_manga


def search_authors_with_works(min_works: int = 3, limit: int = 100) -> list[Author]:
    """Search for authors with at least min_works different manga."""
    print(f"\n🔍 Searching for authors with at least {min_works} works...")
    
    # Find unique authors from popular manga
    print("   Step 1: Finding authors from popular manga...")
    
    discovered_author_ids = set()
    search_configs = [
        {"order[followedCount]": "desc", "offset": offset}
        for offset in [0, 100, 200]
    ] + [
        {"order[rating]": "desc", "offset": offset}
        for offset in [0, 100]
    ] + [{"order[createdAt]": "desc", "offset": 0}]
    
    for config in search_configs:
        for manga in search_manga_api(config):
            for rel in manga.get("relationships", []):
                if rel["type"] == "author":
                    discovered_author_ids.add(rel["id"])
                    break
        print(f"      Found {len(discovered_author_ids)} unique authors so far...")
    
    # Fetch works for each author
    print(f"   Step 2: Fetching works for each author...")
    
    qualified_authors = []
    checked_count = 0
    
    for author_id in discovered_author_ids:
        checked_count += 1
        author_name, works = fetch_author_works(author_id)
        
        if author_name == "Unknown" or len(works) < min_works:
            continue
        
        author = Author(id=author_id, name=author_name)
        author.works = [
            MangaWork(
                id=w["id"], title=w["title"], author_id=author_id,
                author_name=author_name, status=w["status"],
                content_rating=w["content_rating"],
            )
            for w in works
        ]
        
        qualified_authors.append(author)
        print(f"      ✓ {author_name}: {len(works)} works")
        
        if len(qualified_authors) >= limit:
            break
        
        if checked_count % 10 == 0:
            print(f"      Checked {checked_count}/{len(discovered_author_ids)} authors, found {len(qualified_authors)} qualified")
    
    print(f"\n   ✓ Found {len(qualified_authors)} authors with {min_works}+ works")
    return qualified_authors


# ============================================================================
# BUILD FUNCTIONS
# ============================================================================

def download_manga_to_final_dataset(manga: dict, chapters: list[dict]) -> int:
    """Download manga pages to final_dataset with manga_name/chapter structure."""
    manga_dir = FINAL_DATASET_DIR / sanitize_filename(manga["title"])
    manga_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = download_pages_from_chapters(
        chapters, manga_dir, FINAL_PAGES_PER_MANGA,
        FINAL_SKIP_START, FINAL_SKIP_END,
        pages_per_chapter_range=(4, 6)  # ~5 pages per chapter
    )
    
    # Save metadata
    metadata = {
        "id": manga["id"],
        "title": manga["title"],
        "author": manga["author"],
        "status": manga["status"],
        "content_rating": manga["content_rating"],
        "pages_downloaded": downloaded,
    }
    with open(manga_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return downloaded


def download_pages_for_manga(work: MangaWork, output_dir: Path) -> int:
    """Download manga pages to author-based dataset with author/manga/chapter structure."""
    manga_dir = output_dir / sanitize_filename(work.author_name) / sanitize_filename(work.title)
    manga_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "id": work.id,
        "title": work.title,
        "author": work.author_name,
        "author_id": work.author_id,
        "status": work.status,
        "content_rating": work.content_rating,
    }
    with open(manga_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    chapters = get_valid_chapters(work.id)
    if not chapters:
        return 0
    
    return download_pages_from_chapters(
        chapters, manga_dir, PANELS_PER_MANGA,
        PAGES_TO_SKIP_START, PAGES_TO_SKIP_END
    )


def build_final_dataset(num_manga: int = 1, max_search_rounds: int = 10) -> dict:
    """Build final_dataset by downloading random manga until target is reached."""
    print(f"\n{'='*60}")
    print(f"   Building Final Dataset")
    print(f"   Target: {num_manga} manga × {FINAL_PAGES_PER_MANGA} pages each")
    print(f"   Skipping first {FINAL_SKIP_START} and last {FINAL_SKIP_END} pages per chapter")
    print(f"{'='*60}")
    
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {"downloaded_manga": [], "failed_manga": [], "total_pages": 0}
    downloaded_count = 0
    tried_manga_ids = set()
    
    for search_round in range(1, max_search_rounds + 1):
        if downloaded_count >= num_manga:
            break
        
        remaining = num_manga - downloaded_count
        print(f"\n🔄 Search round {search_round}: Need {remaining} more manga...")
        
        candidates = search_random_manga(limit=remaining * 5, exclude_existing=True)
        new_candidates = [m for m in candidates if m["id"] not in tried_manga_ids]
        
        if not new_candidates:
            print("   ⚠️  No new candidates found")
            continue
        
        print(f"   Found {len(new_candidates)} new candidates to try")
        
        for manga in new_candidates:
            if downloaded_count >= num_manga:
                break
            
            tried_manga_ids.add(manga["id"])
            
            is_valid, chapters, _ = validate_manga_for_final_dataset(manga)
            if not is_valid:
                stats["failed_manga"].append({"title": manga["title"], "reason": "Validation failed"})
                continue
            
            print(f"   📥 Downloading: {manga['title'][:50]}...")
            pages = download_manga_to_final_dataset(manga, chapters)
            
            if pages >= FINAL_PAGES_PER_MANGA:
                downloaded_count += 1
                print(f"      ✅ Downloaded {pages} pages ({downloaded_count}/{num_manga})")
                stats["downloaded_manga"].append({
                    "title": manga["title"], "author": manga["author"], "pages": pages
                })
                stats["total_pages"] += pages
            else:
                print(f"      ❌ Only got {pages}/{FINAL_PAGES_PER_MANGA} pages")
                stats["failed_manga"].append({
                    "title": manga["title"], "reason": f"Only {pages} pages"
                })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"   Final Dataset Summary")
    print(f"{'='*60}")
    print(f"   Manga tried: {len(tried_manga_ids)}")
    print(f"   Downloaded: {len(stats['downloaded_manga'])}/{num_manga} manga")
    print(f"   Total pages: {stats['total_pages']}")
    
    if stats["downloaded_manga"]:
        print(f"\n   ✅ Successfully downloaded:")
        for m in stats["downloaded_manga"]:
            print(f"      - {m['title'][:40]} ({m['pages']} pages)")
    
    if len(stats['downloaded_manga']) < num_manga:
        print(f"\n   ⚠️  Could not reach target of {num_manga} manga")
        if stats["failed_manga"]:
            print(f"   Recent failures:")
            for m in stats["failed_manga"][-5:]:
                print(f"      - {m['title'][:40]}: {m['reason']}")
    
    print(f"\n   Saved to: {FINAL_DATASET_DIR}")
    return stats


def select_authors_for_dataset(authors: list[Author], num_authors: int) -> list[Author]:
    """Select and validate authors for dataset building."""
    print(f"\n📋 Selecting {num_authors} authors (each with {WORKS_PER_AUTHOR} manga)...")
    
    selected = []
    random.shuffle(authors)
    
    for author in authors:
        if len(selected) >= num_authors:
            break
        
        print(f"\n   Checking author: {author.name} ({len(author.works)} works available)")
        
        valid_works = []
        for work in author.works:
            if validate_manga_for_dataset(work):
                valid_works.append(work)
                print(f"      ✓ Valid: {work.title[:40]}")
            else:
                print(f"      ✗ Invalid: {work.title[:40]}")
            
            if len(valid_works) >= WORKS_PER_AUTHOR:
                break
            rate_limit_sleep(0.3)
        
        if len(valid_works) >= WORKS_PER_AUTHOR:
            author.works = valid_works[:WORKS_PER_AUTHOR]
            selected.append(author)
            print(f"   ✅ Selected {author.name} with {WORKS_PER_AUTHOR} works")
        else:
            print(f"   ❌ {author.name} only has {len(valid_works)}/{WORKS_PER_AUTHOR} valid works - SKIPPING")
    
    if len(selected) < num_authors:
        print(f"\n   ⚠️  WARNING: Only found {len(selected)}/{num_authors} qualified authors!")
    
    return selected


def build_dataset(authors: list[Author], dataset_name: str, output_dir: Path) -> dict:
    """Build author-based dataset by downloading pages for all works."""
    print(f"\n{'='*60}")
    print(f"   Building Dataset: {dataset_name}")
    print(f"   Target: {len(authors)} authors × {WORKS_PER_AUTHOR} manga × {PANELS_PER_MANGA} pages")
    print(f"{'='*60}")
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "name": dataset_name, "authors": [],
        "total_mangas": 0, "total_pages": 0,
        "complete_mangas": 0, "incomplete_mangas": []
    }
    
    for author in authors:
        print(f"\n📚 Processing author: {author.name}")
        author_stats = {"name": author.name, "id": author.id, "works": []}
        
        for work in author.works:
            print(f"   📖 Downloading: {work.title[:50]}...")
            pages = download_pages_for_manga(work, dataset_dir)
            
            if pages >= PANELS_PER_MANGA:
                print(f"      ✅ Downloaded {pages} pages")
                stats["complete_mangas"] += 1
            else:
                print(f"      ❌ Only got {pages}/{PANELS_PER_MANGA} pages")
                stats["incomplete_mangas"].append({
                    "author": author.name, "title": work.title, "pages": pages
                })
            
            author_stats["works"].append({
                "title": work.title, "id": work.id, "pages_downloaded": pages
            })
            stats["total_pages"] += pages
            stats["total_mangas"] += 1
        
        stats["authors"].append(author_stats)
    
    # Save manifest
    with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Summary
    expected = len(authors) * WORKS_PER_AUTHOR
    print(f"\n{'='*60}")
    print(f"   Dataset '{dataset_name}' Summary")
    print(f"{'='*60}")
    print(f"   Authors: {len(authors)}")
    print(f"   Mangas: {stats['total_mangas']} | Complete: {stats['complete_mangas']}")
    print(f"   Pages: {stats['total_pages']} (target: {expected * PANELS_PER_MANGA})")
    print(f"\n   Saved to: {dataset_dir}")
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for dataset building."""
    print("\n" + "=" * 60)
    print("       MANGA DATASET BUILDER")
    print("=" * 60)
    print(f"\nModes available:")
    print(f"  • Final Dataset: Random manga to final_dataset/")
    print(f"    - {FINAL_PAGES_PER_MANGA} pages per manga, skip first/last {FINAL_SKIP_START}/{FINAL_SKIP_END}")
    print(f"\n  • Author-based datasets:")
    print(f"    - Small: 9 mangas | Medium: 21 | Large: 51")
    print(f"    - {PANELS_PER_MANGA} pages per manga\n")
    
    DATASETS = [
        {"name": "small", "num_authors": 3},
        {"name": "medium", "num_authors": 7},
        {"name": "large", "num_authors": 17},
    ]
    
    all_authors = None
    
    while True:
        print("\n" + "-" * 40)
        print("[Menu]")
        print("  1. Download to final_dataset (random manga)")
        print("  2. Build small dataset (3 authors)")
        print("  3. Build medium dataset (7 authors)")
        print("  4. Build large dataset (17 authors)")
        print("  5. Build all author-based datasets")
        print("  6. Show qualified authors")
        print("  0. Exit")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        elif choice == "1":
            existing = get_existing_manga_in_final_dataset()
            print(f"\n📂 Current final_dataset has {len(existing)} manga")
            
            try:
                num_manga = int(input("How many manga? [1]: ").strip() or "1")
                num_manga = max(1, num_manga)
            except ValueError:
                num_manga = 1
            
            build_final_dataset(num_manga=num_manga)
        
        elif choice == "6":
            if all_authors is None:
                all_authors = search_authors_with_works(min_works=WORKS_PER_AUTHOR, limit=100)
            
            print(f"\n📋 Qualified Authors ({len(all_authors)} total):")
            for i, author in enumerate(all_authors[:30], 1):
                works = ", ".join(w.title[:25] for w in author.works[:3])
                print(f"  {i:2}. {author.name}: {works}...")
        
        elif choice in ["2", "3", "4", "5"]:
            if all_authors is None:
                all_authors = search_authors_with_works(min_works=WORKS_PER_AUTHOR, limit=100)
            
            datasets_to_build = {
                "2": [DATASETS[0]],
                "3": [DATASETS[1]],
                "4": [DATASETS[2]],
                "5": DATASETS
            }[choice]
            
            max_authors = max(d["num_authors"] for d in datasets_to_build)
            selected = select_authors_for_dataset(all_authors, max_authors)
            
            if len(selected) < max_authors:
                if input(f"Only {len(selected)} authors. Continue? (y/n): ").strip().lower() != "y":
                    continue
            
            for ds in datasets_to_build:
                authors = selected[:ds["num_authors"]]
                build_dataset(authors, ds["name"], DATASET_DIR)
            
            print("\n" + "=" * 60)
            print("   ALL DATASETS COMPLETE!")
            print("=" * 60)
        
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
