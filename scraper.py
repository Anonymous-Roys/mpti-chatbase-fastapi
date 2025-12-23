import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import json
from typing import List, Dict, Any, Optional

RATE_LIMIT_DELAY = 5  # 1 second between requests
MAX_PAGES = 100
OUTPUT_FILE = "scraped_data.json"

def clean_text(soup: BeautifulSoup) -> str:
    """Extracts and cleans text content, removing boilerplate elements."""
    # Elements to remove (navigation, footer, scripts, styles)
    elements_to_remove = [
        'nav', 'footer', 'script', 'style', 'header',
        '.sidebar', '#sidebar', '.nav', '.footer', 'form',
        'noscript', 'img', 'svg', 'iframe'
    ]
    for element in soup.find_all(elements_to_remove):
        element.decompose()

    # Get the main content area (heuristically)
    main_content = soup.find('body')
    if main_content:
        # Get text, strip excess whitespace and newlines
        text = main_content.get_text(separator=' ', strip=True)
        return text
    return ""

def is_valid(url: str, base_url: str) -> bool:
    """Checks if a URL is valid and internal to the base domain."""
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)
    return bool(parsed.netloc) and parsed.netloc == base_parsed.netloc

def scrape_website(base_url: str, max_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
    """
    Performs a BFS crawl of the website starting from the base_url.
    """
    print(f"Starting crawl from: {base_url} (Max pages: {max_pages})")
    queue = [base_url]
    visited = {base_url}
    scraped_data: List[Dict[str, Any]] = []
    
    count = 0

    while queue and count < max_pages:
        current_url = queue.pop(0)
        print(f"[{count + 1}/{max_pages}] Fetching: {current_url}")
        
        try:
            # Send GET request
            response = requests.get(current_url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. Extract and Clean Content
            content = clean_text(soup)
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No Title"

            scraped_data.append({
                "url": current_url,
                "title": title,
                "content": content,
                "length": len(content)
            })
            count += 1
            
            # 2. Find internal links for BFS
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                absolute_url = urljoin(current_url, href)
                
                if is_valid(absolute_url, base_url) and absolute_url not in visited and absolute_url not in queue:
                    # Ignore anchor links and specific file extensions
                    if '#' not in absolute_url and not any(absolute_url.endswith(ext) for ext in ['.pdf', '.zip', '.jpg', '.png']):
                        queue.append(absolute_url)
                        visited.add(absolute_url)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    print(f"Crawl finished. Scraped {len(scraped_data)} pages.")
    return scraped_data

def save_data(data: List[Dict[str, Any]]):
    """Saves the scraped data to a JSON file."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Example usage: Replace with the target website URL
    target_url = input("Enter the base URL to scrape (e.g., https://example.com): ")
    if not target_url:
        target_url = "https://www.mptigh.com/" # Default for MPTI Ghana website

    scraped_results = scrape_website(target_url)
    if scraped_results:
        save_data(scraped_results)