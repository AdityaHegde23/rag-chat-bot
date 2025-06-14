import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
import json

class AngelOneScraper:
    def __init__(self):
        self.base_url = "https://www.angelone.in/support"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.scraped_data = []
        
    def get_page_content(self, url):
        """Extract text content from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get page title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.page-content', 'body'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    content_text = content.get_text(separator='\n', strip=True)
                    break
            
            if not content_text:
                content_text = soup.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = content_text.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            content_text = '\n'.join(cleaned_lines)
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': "",
                'content': "",
                'status': 'error',
                'error': str(e)
            }
    
    def find_support_links(self):
        """Find all support page links from the main support page"""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links that seem to be support-related
            links = set()
            
            # Look for links within the support domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.base_url, href)
                
                # Only include links that are within the support section
                if 'angelone.in/support' in full_url and full_url != self.base_url:
                    links.add(full_url)
            
            return list(links)
            
        except Exception as e:
            print(f"Error finding support links: {str(e)}")
            return []
    
    def scrape_all_pages(self):
        """Scrape all support pages"""
        print("Finding support page links...")
        
        # Start with the main support page
        all_urls = [self.base_url]
        
        # Find additional support links
        support_links = self.find_support_links()
        all_urls.extend(support_links)
        
        print(f"Found {len(all_urls)} URLs to scrape")
        
        # Scrape each page
        for i, url in enumerate(all_urls, 1):
            print(f"Scraping {i}/{len(all_urls)}: {url}")
            
            page_data = self.get_page_content(url)
            if page_data['status'] == 'success' and page_data['content']:
                self.scraped_data.append(page_data)
                print(f"Successfully scraped: {page_data['title']}")
            else:
                print(f"Failed to scrape: {url}")
            
            time.sleep(1)
        
        print(f"\nScraping completed! Successfully scraped {len(self.scraped_data)} pages")
        return self.scraped_data
    
    def save_to_files(self, output_dir="scraped_data"):
        """Save scraped data to text files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save individual text files
        for i, page_data in enumerate(self.scraped_data):
            filename = f"page_{i+1}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {page_data['url']}\n")
                f.write(f"Title: {page_data['title']}\n")
                f.write(f"{'='*50}\n\n")
                f.write(page_data['content'])
        
        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.scraped_data)} files to {output_dir}/")

def main():
    """Main function to run the scraper"""
    scraper = AngelOneScraper()
    
    print("Starting Scraper...")
    
    # Scrape all pages
    scraped_data = scraper.scrape_all_pages()
    
    if scraped_data:
        # Save to files
        scraper.save_to_files()
        print("\nScraping completed successfully!")
    else:
        print("No data was scraped. Please check the website URL and try again.")

if __name__ == "__main__":
    main()