# code adapted from
# https://python.plainenglish.io/how-to-scrape-images-using-beautifulsoup4-in-python-e7a4ddb904b8
# https://serpapi.com/blog/scrape-google-images-with-python/


import os
import requests
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm


def get_url(query):
    '''
    Get url og Google results page we want to scrape for images
    
    :param query: str of query word to search in Google
    
    :returns: str url of images results page on Google
    '''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                       AppleWebKit/537.36 (KHTML, like Gecko) \
                       Chrome/103.0.5060.114 \
                       Safari/537.36"
    }
    params = {
        "q": query,      # search query
        "tbm": "isch",   # image results
        "hl": "en",      # language of the search
        "gl": "en",      # country where search comes from
        "ijn": "0"       # page number
    }
    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=5)
    return html.url


def scrape_links(url):
    '''
    Scrape links for individual images on the results page
    Note that auto scraping of google is limited to initial 20 results
    To scrape for more results another package such as Selenium should be used
    
    :param url: str url of Google images results page
    
    :returns: list of strs of urls for individual images
    '''
    # set html content of google images tab to the variable "page"
    page = requests.get(url)
    
    # create soup object with page contents
    soup = BeautifulSoup(page.content, "html.parser")
    
    # yWs4tf is the default class every img tag in the images section of Google has
    image_tags = soup.find_all("img", class_="yWs4tf")
    links = [image_tag['src'] for image_tag in image_tags]

    return links


def download_images(links, save_dir):
    '''
    Download images and save these locally for training our network on later
    tqdm is used to show progress bar of these downloads
    
    :param links: list of strs of urls for individual images
    :param save_dir: str full path to directory to save images in
    '''
    print(f"\nDownloading {len(links)} images for {save_dir.split('/')[-1]} dir...")
    for idx, link in tqdm(enumerate(links)):
        urllib.request.urlretrieve(link, os.path.join(save_dir, f"img_{idx}.jpg"))
        

def main():
    '''
    Download images for search queries 'butter' and 'margarine'
    '''
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(root_dir, "data")
            
    for query in ["butter", "margarine"]:
        
        directory = os.path.join(data_dir, query)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        url = get_url(query)
        links = scrape_links(url)
        download_images(links, directory)

    
if __name__ == "__main__":
    
    main()
