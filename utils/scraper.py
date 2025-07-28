# utils/scraper.py
from icrawler.builtin import GoogleImageCrawler

def download_celebrity_images(celebrity_name, num_images=10):
    crawler = GoogleImageCrawler(storage={'root_dir': f'hollywood_faces/{celebrity_name}'})
    crawler.crawl(keyword=celebrity_name, max_num=num_images)

# Example usage:
# download_celebrity_images("Brad Pitt")
