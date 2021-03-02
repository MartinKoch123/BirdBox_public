from pathlib import Path
import hashlib
import time
from datetime import datetime
import requests
from typing import Union
import io
from math import inf
from PIL import Image as PIL_Image
from PIL import UnidentifiedImageError
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException, StaleElementReferenceException, \
    ElementClickInterceptedException, NoSuchElementException
from progressbar import progressbar
from peewee import IntegrityError
from birdbox.tools import StopFile, insane
from birdbox.database import Image, Url, SearchHit, Search

DRIVER_PATH = Path(r"chromedriver.exe")
WAIT_TIME = 0.1


def create_hash(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]


def search_macaulay(search_term_entry, limit=inf):

    search_url = f"https://search.macaulaylibrary.org/catalog?taxonCode={search_term_entry.term}"

    with webdriver.Chrome(executable_path=str(DRIVER_PATH)) as wd:
        wd.get(search_url)

        press_photos_only_filter_button(wd)
        press_more_button_repeatedly(wd, limit=limit)

        elements = wd.find_elements_by_css_selector("img")

        new_urls_counter = 0
        for element in progressbar(elements):
            url_list = element.get_attribute("srcset")
            url = url_list.split(",")[-1]
            url = url.split(" ")[0]
            if url == "":
                continue

            new_urls_counter += int(store_search_hit(url, search_term_entry))

        Search(error_count=0, hit_count=new_urls_counter, search_term=search_term_entry,
               time=datetime.now()).save()


def press_photos_only_filter_button(wd):
    photos_only_button = [button for button in wd.find_elements_by_class_name("RadioGroup-secondary")
                          if button.get_attribute("data-media-type-count") == "photo"][0]
    photos_only_button.click()
    time.sleep(1)


def press_more_button_repeatedly(wd, limit=inf):
    counter = 0
    try:
        while counter < limit:
            macaulay_press_more_button(wd)
            time.sleep(1)
            counter += 1
    except (ElementNotInteractableException, ElementClickInterceptedException, NoSuchElementException):
        pass


@insane(errors=(ElementNotInteractableException, ElementClickInterceptedException, NoSuchElementException), wait_periods=(0.3, 1, 3, 10))
def macaulay_press_more_button(wd):
    more_button = wd.find_element_by_id("show_more")
    more_button.click()

def search(search_term_entry):

    raise NotImplementedError("Implement width and height of image")

    search_url = f"https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={search_term_entry.term}&oq={search_term_entry.term}&gs_l=img"
    stop_file = StopFile("stop.txt", 10)

    with webdriver.Chrome(executable_path=str(DRIVER_PATH)) as wd:
        wd.get(search_url)

        new_urls_counter, error_counter, inspected_thumbnail_counter, thumbnail_count = 0, 0, 0, 0
        while True:
            scroll_to_end(wd)
            thumbnails = wd.find_elements_by_css_selector("img.Q4LuWd")
            if len(thumbnails) == inspected_thumbnail_counter:

                press_more_button(wd)
                thumbnails = wd.find_elements_by_css_selector("img.Q4LuWd")
                if len(thumbnails) == inspected_thumbnail_counter:
                    break

            while inspected_thumbnail_counter < len(thumbnails):
                inspected_thumbnail_counter += 1
                try:
                    urls = extract_urls_from_thumbnail(wd, thumbnails[inspected_thumbnail_counter-1])
                except (CouldNotExtractUrlError, StaleElementReferenceException, ElementClickInterceptedException, ElementNotInteractableException):
                    error_counter += 1
                    continue                
                if stop_file.stop_is_pending():
                    return
                new_urls_counter += sum([store_search_hit(url, search_term_entry) for url in urls])
    Search(error_count=error_counter, hit_count=new_urls_counter, search_term=search_term_entry, time=datetime.now()).save()


def store_search_hit(url, search_term_entry):
    url_query = Url.select().where(Url.url == url)
    if url_query.exists():
        url_entry = list(url_query)[0]
    else:
        url_entry = Url(url=url)
        url_entry.save()
    if SearchHit.select().where(SearchHit.url == url_entry, SearchHit.search_term == search_term_entry).exists():
        return False
    search_hit_entry = SearchHit(url=url_entry, search_term=search_term_entry)
    search_hit_entry.save()
    return True


def download(folder_path: Union[Path, str], url_entry: Url):
    if url_entry.image_id:
        raise Exception("Url is already associated with an image.")
    try:
        image_content = requests.get(url_entry.url).content
    except (requests.exceptions.ConnectionError, requests.exceptions.InvalidSchema):
        url_entry.is_broken = True
        url_entry.save()
        return
    image_hash = create_hash(image_content)
    image_entry = Image(name=image_hash)

    image_bytes = io.BytesIO(image_content)
    try:
        image = PIL_Image.open(image_bytes).convert('RGB')
    except UnidentifiedImageError:
        url_entry.is_broken = True
        url_entry.save()
        return
    image_entry.width = image.size[0]
    image_entry.height = image.size[1]

    image_path = Path(folder_path) / (image_hash + '.jpg')
    with open(image_path, 'wb') as f:
        image.save(f, "JPEG", quality=100)
    try:
        image_entry.save()
    except IntegrityError:
        url_entry.image = Image.get(Image.name == image_hash)
    else:
        url_entry.image = image_entry
    url_entry.save()


def press_more_button(wd):
    wd.find_element_by_css_selector(".mye4qd")
    wd.execute_script("document.querySelector('.mye4qd').click();")
    time.sleep(1)


def scroll_to_end(wd):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.2)


class CouldNotExtractUrlError(Exception):
    pass


def extract_urls_from_thumbnail(wd, thumbnail):
    try_click(thumbnail)
    for i in range(100):
        urls = [image.get_attribute("src") for image in wd.find_elements_by_css_selector('img.n3VNCb')]
        urls = set([url for url in urls if url and "http" in url])
        if len(urls) > 0:
            return urls
        time.sleep(0.01)
    raise CouldNotExtractUrlError("Could not extract url.")


def try_click(img):
    for i_try in range(5):
        try:
            time.sleep(WAIT_TIME * 2 ** i_try)
            img.click()
            return
        except StaleElementReferenceException as e:
            raise e
        except Exception as e:
            if i_try == 4:
                raise e

