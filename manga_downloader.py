import requests
import os
import json
import random
from pathlib import Path

# API endpoints
BASE_URL = "https://api.mangadex.org"
MANGA_URL = f"{BASE_URL}/manga"
CHAPTER_URL = f"{BASE_URL}/chapter"
PAGE_URL = f"{BASE_URL}/at-home/server/"

HEADERS = {
    "User-Agent": "MyMangaApp/1.0 (contact@example.com)"
}

# Download directory
DOWNLOAD_DIR = Path("./downloads")


def search_manga(query: str, search_by: str = "title", limit: int = 10) -> list:
    """
    Search for manga by title or author name.
    Returns a list of manga with their details.
    """
    author_cache = {}  # Cache author names by ID
    
    if search_by == "author":
        # First search for author
        author_resp = requests.get(
            f"{BASE_URL}/author",
            params={"name": query, "limit": 5},
            headers=HEADERS
        )
        author_data = author_resp.json()
        
        if not author_data.get("data"):
            return []
        
        # Cache author names and get IDs
        author_ids = []
        for author in author_data["data"]:
            author_ids.append(author["id"])
            author_cache[author["id"]] = author["attributes"].get("name", "Unknown")
        
        params = {
            "authors[]": author_ids,
            "limit": limit,
            "includes[]": ["cover_art", "author"]
        }
    else:
        params = {
            "title": query,
            "limit": limit,
            "includes[]": ["cover_art", "author"]
        }
    
    resp = requests.get(MANGA_URL, params=params, headers=HEADERS)
    data = resp.json()
    
    results = []
    for manga in data.get("data", []):
        # Get title (prefer English)
        titles = manga["attributes"]["title"]
        title = titles.get("en") or titles.get("ja-ro") or list(titles.values())[0] if titles else "Unknown"
        
        # Get author name from relationships or cache
        author_name = "Unknown"
        author_id = None
        for rel in manga.get("relationships", []):
            if rel["type"] == "author":
                author_id = rel["id"]
                if "attributes" in rel:
                    author_name = rel["attributes"].get("name", "Unknown")
                elif author_id in author_cache:
                    author_name = author_cache[author_id]
                break
        
        # If still unknown, fetch author details
        if author_name == "Unknown" and author_id:
            try:
                author_resp = requests.get(f"{BASE_URL}/author/{author_id}", headers=HEADERS)
                author_data = author_resp.json()
                if author_data.get("data"):
                    author_name = author_data["data"]["attributes"].get("name", "Unknown")
            except:
                pass
        
        # Get publication status
        status = manga["attributes"].get("status", "unknown")
        publication_demographic = manga["attributes"].get("publicationDemographic", "N/A")
        
        # Get description (prefer English)
        descriptions = manga["attributes"].get("description", {})
        description = descriptions.get("en") or next(iter(descriptions.values()), "No description available.")
        
        # Get genres/tags
        tags = manga["attributes"].get("tags", [])
        genres = [
            tag["attributes"]["name"].get("en", tag["attributes"]["name"].get("ja-ro", ""))
            for tag in tags
            if tag["attributes"].get("group") == "genre"
        ]
        themes = [
            tag["attributes"]["name"].get("en", tag["attributes"]["name"].get("ja-ro", ""))
            for tag in tags
            if tag["attributes"].get("group") == "theme"
        ]
        
        results.append({
            "id": manga["id"],
            "title": title,
            "author": author_name,
            "status": status,
            "demographic": publication_demographic,
            "year": manga["attributes"].get("year"),
            "content_rating": manga["attributes"].get("contentRating", "unknown"),
            "description": description,
            "genres": genres,
            "themes": themes
        })
    
    return results


def get_chapter_info(manga_id: str) -> dict:
    """
    Get chapter information for a manga.
    Returns min/max chapter numbers and total count.
    """
    # Get total chapter count and aggregate info
    aggregate_resp = requests.get(
        f"{MANGA_URL}/{manga_id}/aggregate",
        params={"translatedLanguage[]": ["en"]},
        headers=HEADERS
    )
    aggregate_data = aggregate_resp.json()
    
    chapters = []
    volumes = aggregate_data.get("volumes", {})
    
    for vol_key, vol_data in volumes.items():
        for ch_key, ch_data in vol_data.get("chapters", {}).items():
            try:
                ch_num = float(ch_key) if ch_key != "none" else 0
                chapters.append({
                    "chapter": ch_key,
                    "chapter_num": ch_num,
                    "id": ch_data.get("id"),
                    "count": ch_data.get("count", 1)
                })
            except ValueError:
                continue
    
    if not chapters:
        return {"min": None, "max": None, "total": 0, "chapters": []}
    
    chapters.sort(key=lambda x: x["chapter_num"])
    
    return {
        "min": chapters[0]["chapter"],
        "max": chapters[-1]["chapter"],
        "total": len(chapters),
        "chapters": chapters
    }


def get_chapter_id(manga_id: str, chapter_number: str) -> str | None:
    """
    Get the chapter ID for a specific chapter number.
    """
    params = {
        "manga": manga_id,
        "chapter": chapter_number,
        "translatedLanguage[]": ["en"],
        "limit": 1
    }
    
    resp = requests.get(CHAPTER_URL, params=params, headers=HEADERS)
    data = resp.json()
    
    if data.get("data"):
        return data["data"][0]["id"]
    return None


def select_human_name(author_name: str) -> str:
    """
    Select the most human-like name from potentially multiple authors.
    Prefers names with first and last name pattern (contains space).
    """
    if not author_name or author_name == "Unknown":
        return "Unknown Author"
    
    # If multiple authors separated by comma or &
    authors = [a.strip() for a in author_name.replace(" & ", ", ").split(",")]
    
    # Filter for names that look like human names (contain a space = first + last name)
    human_names = [a for a in authors if " " in a and not a.isupper()]
    
    if human_names:
        return human_names[0]
    
    # Otherwise return the first author
    return authors[0]


def save_manga_metadata(manga_dir: Path, manga: dict) -> None:
    """
    Save manga metadata to a JSON file in the manga folder.
    """
    metadata_file = manga_dir / "metadata.json"
    
    # Only write if doesn't exist or we want to update
    metadata = {
        "title": manga.get("title", "Unknown"),
        "author": manga.get("author", "Unknown"),
        "status": manga.get("status", "unknown"),
        "year": manga.get("year"),
        "content_rating": manga.get("content_rating", "unknown"),
        "demographic": manga.get("demographic", "N/A"),
        "genres": manga.get("genres", []),
        "themes": manga.get("themes", []),
        "description": manga.get("description", "No description available.")
    }
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Metadata saved to: {metadata_file}")


def download_chapter_images(chapter_id: str, manga_title: str, chapter_number: str, manga: dict) -> bool:
    """
    Download all images for a chapter.
    Saves to: downloads/{author_name}/{manga_title}/chapter_{number}/
    Also creates metadata.json in the manga folder.
    """
    resp = requests.get(PAGE_URL + chapter_id, headers=HEADERS)
    server_data = resp.json()
    
    if server_data.get("result") == "error":
        print(f"Error: Could not get chapter data")
        return False
    
    base_url = server_data["baseUrl"]
    chapter_hash = server_data["chapter"]["hash"]
    pages = server_data["chapter"]["data"]
    
    # Create download directory with author/manga structure
    author_name = manga.get("author", "Unknown")
    selected_author = select_human_name(author_name)
    safe_author = "".join(c for c in selected_author if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = "".join(c for c in manga_title if c.isalnum() or c in (' ', '-', '_')).strip()
    
    # Debug: show what author name is being used
    print(f"\nAuthor (raw): {author_name}")
    print(f"Author (selected): {selected_author}")
    
    # Create directories
    manga_dir = DOWNLOAD_DIR / safe_author / safe_title
    chapter_dir = manga_dir / f"chapter_{chapter_number}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata to manga folder (creates/updates metadata.json)
    save_manga_metadata(manga_dir, manga)
    
    # Select 5 random pages (or all if fewer than 5)
    num_to_download = min(5, len(pages))
    selected_indices = sorted(random.sample(range(len(pages)), num_to_download))
    selected_pages = [(idx + 1, pages[idx]) for idx in selected_indices]
    
    print(f"Downloading {num_to_download} random pages (out of {len(pages)}) to: {chapter_dir}")
    
    for count, (page_num, page) in enumerate(selected_pages, 1):
        image_url = f"{base_url}/data/{chapter_hash}/{page}"
        
        try:
            img_resp = requests.get(image_url, headers=HEADERS)
            img_resp.raise_for_status()
            
            # Get file extension from page name
            ext = Path(page).suffix or ".jpg"
            filename = f"page_{page_num:03d}{ext}"
            filepath = chapter_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            
            print(f"  Downloaded page {page_num} ({count}/{num_to_download})", end="\r")
        except Exception as e:
            print(f"\n  Error downloading page {page_num}: {e}")
    
    print(f"\n✓ Downloaded {num_to_download} random pages successfully!")
    return True


def display_manga_list(manga_list: list) -> None:
    """Display manga search results."""
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    
    for i, manga in enumerate(manga_list, 1):
        year = f" ({manga['year']})" if manga['year'] else ""
        print(f"\n[{i}] {manga['title']}{year}")
        print(f"    Author: {manga['author']}")
        print(f"    Status: {manga['status'].capitalize()}")
        print(f"    Rating: {manga['content_rating']}")
    
    print("\n" + "-" * 60)


def display_chapter_info(manga: dict, chapter_info: dict) -> None:
    """Display chapter information for selected manga."""
    print("\n" + "=" * 60)
    print(f"MANGA: {manga['title']}")
    print("=" * 60)
    
    # Publication status
    is_official = manga['status'] == 'completed'
    status_icon = "✓" if is_official else "○"
    print(f"\n{status_icon} Publication Status: {manga['status'].capitalize()}")
    print(f"  Content Rating: {manga['content_rating']}")
    
    if manga['demographic']:
        print(f"  Demographic: {manga['demographic']}")
    
    print(f"\n📚 Available Chapters (English):")
    print(f"   Min Chapter: {chapter_info['min']}")
    print(f"   Max Chapter: {chapter_info['max']}")
    print(f"   Total: {chapter_info['total']} chapters")
    
    print("\n" + "-" * 60)


def main():
    """Main pipeline loop."""
    print("\n" + "=" * 60)
    print("       MANGADEX DOWNLOADER")
    print("=" * 60)
    
    while True:
        # Step 1: Search
        print("\n[Search for Manga]")
        print("  1. Search by title")
        print("  2. Search by author")
        print("  0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        if choice not in ["1", "2"]:
            print("Invalid option. Please try again.")
            continue
        
        search_type = "title" if choice == "1" else "author"
        query = input(f"\nEnter {search_type} name (partial is OK): ").strip()
        
        if not query:
            print("Search query cannot be empty.")
            continue
        
        print(f"\nSearching for '{query}' by {search_type}...")
        manga_list = search_manga(query, search_by=search_type)
        
        if not manga_list:
            print("No manga found. Try a different search term.")
            continue
        
        # Step 2: Select manga
        while True:
            display_manga_list(manga_list)
            print("  [0] Go back to search")
            
            selection = input("\nSelect manga number: ").strip()
            
            if selection == "0":
                break
            
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(manga_list):
                    selected_manga = manga_list[idx]
                else:
                    print("Invalid selection. Please try again.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue
            
            # Step 3: Show chapter info
            print(f"\nFetching chapter info for '{selected_manga['title']}'...")
            chapter_info = get_chapter_info(selected_manga['id'])
            
            if chapter_info['total'] == 0:
                print("No English chapters available for this manga.")
                continue
            
            while True:
                display_chapter_info(selected_manga, chapter_info)
                print("  Enter chapter number to download")
                print("  [0] Go back to manga selection")
                
                ch_input = input("\nEnter chapter number: ").strip()
                
                if ch_input == "0":
                    break
                
                # Validate chapter number
                try:
                    ch_num = float(ch_input)
                    min_ch = float(chapter_info['min'])
                    max_ch = float(chapter_info['max'])
                    
                    if ch_num < min_ch or ch_num > max_ch:
                        print(f"Chapter must be between {chapter_info['min']} and {chapter_info['max']}")
                        continue
                except ValueError:
                    print("Please enter a valid chapter number.")
                    continue
                
                # Step 4: Download chapter
                print(f"\nLooking for chapter {ch_input}...")
                chapter_id = get_chapter_id(selected_manga['id'], ch_input)
                
                if not chapter_id:
                    print(f"Could not find chapter {ch_input}. It may not be available in English.")
                    continue
                
                download_chapter_images(chapter_id, selected_manga['title'], ch_input, selected_manga)
                
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
