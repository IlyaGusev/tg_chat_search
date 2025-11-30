import json
import shutil
from datetime import datetime, timezone, timedelta
from typing import List

import fire
import requests
import html2text
from lxml import etree
from lxml.html import tostring
from bs4 import BeautifulSoup


def get_current_ts():
    return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())


def parse_post_url(url):
    url = url.split("?")[0]
    channel_id, post_id = url.split("/")[-2:]
    return {
        "url": url,
        "channel_id": channel_id.lower(),
        "post_id": int(post_id),
        "id": int(post_id),
    }


def to_timestamp(dt_str):
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S+00:00")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def html2text_setup():
    instance = html2text.HTML2Text(bodywidth=0)
    instance.ignore_links = True
    instance.ignore_images = True
    instance.ignore_tables = True
    instance.ignore_emphasis = True
    instance.ul_item_mark = ""
    return instance


class TelegramSpider:
    name = "telegram"
    channel_url_template = "https://t.me/s/{}"
    post_url_template = "https://t.me/{}?embed=1"

    def __init__(self, channels: List[str], *args, **kwargs):
        self.channels = channels
        self.html2text = html2text_setup()

        super().__init__(*args, **kwargs)

    def __call__(self):
        urls = {self.channel_url_template.format(ch) for ch in self.channels}
        records = []
        for url in urls:
            records += self.parse_channel(url)
        return records

    def parse_channel(self, url):
        print(url)
        channel_name = url.split("/")[-1].split("?")[0]
        history_path = "//body/main/div/section[contains(@class, 'tgme_channel_history')]/div"
        doc = requests.get(url).content
        response = etree.HTML(doc)
        soup = BeautifulSoup(doc, 'html.parser')
        posts = response.xpath(history_path + "/div")

        records = []
        min_post_id = None
        for post in posts:
            post_path = post.xpath("@data-post")[0]
            post_soup = BeautifulSoup(tostring(post), features="lxml")
            post_time_element = post_soup.select_one("time.time")
            post_time = post_time_element.get('datetime') if post_time_element else None
            if not post_path or not post_time:
                continue

            post_id = int(post_path.split("/")[-1])
            post_ts = to_timestamp(post_time)
            min_post_id = min(post_id, min_post_id) if min_post_id is not None else post_id
            post_url = self.post_url_template.format(post_path)
            try:
                item = self._parse_post(post_soup, post_url)
                if item is None:
                    continue
                if post_id <= 2:
                    continue
                item["source"] = channel_name
                records.append(item)
            except Exception as e:
                print(f"Unexpected error at {post_url}:", str(e))
                continue

        current_ts = get_current_ts()
        url = url.split("?")[0]
        url += "?before={}".format(min_post_id)
        if min_post_id <= 1:
            return records
        return records + self.parse_channel(url)

    def _parse_post(self, post_element, post_url):
        text_path = "div.tgme_widget_message_bubble > div.tgme_widget_message_text"
        text_alt_path = "div.tgme_widget_message_bubble > div.media_supported_cont > div.tgme_widget_message_text"
        time_path = "time.time"
        reply_path = "a.tgme_widget_message_reply"
        forward_path = "a.tgme_widget_message_forwarded_from_name"

        item = parse_post_url(post_url)
        text_element = post_element.select_one(text_path)
        text_alt_element = post_element.select_one(text_alt_path)
        if not text_element and text_alt_element:
            text_element = text_alt_element

        if not text_element:
            # Images only
            return None

        item["text"] = self._parse_html(str(text_element))
        item["links"] = [l.get('href') for l in text_element.select("a")]
        item["fetch_time"] = int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

        time_element = post_element.select_one(time_path).get("datetime")
        item["pub_time"] = to_timestamp(time_element)

        reply_element = post_element.select_one(reply_path)
        if reply_element:
            item["reply_to"] = reply_element.get("href")
            item["reply_to_message_id"] = item["reply_to"].split("/")[-1]

        forward_element = post_element.select_one(forward_path)
        if forward_element:
            return None

        return item

    def _parse_html(self, html):
        text = self.html2text.handle(html)
        sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
        for i, sentence in enumerate(sentences):
            if sentence[-1].isalpha():
                sentences[i] = sentence + "."
        return "\n".join(sentences)


def download_channels(output_file: str) -> None:
    spider = TelegramSpider([
        "senior_augur",
        "denissexy",
        "doomgrad",
        "seeallochnaya",
        "izolenta_mebiusa",
        "tech_priestess",
        "lovedeathtransformers",
        "knowledge_accumulator",
        "vikhrlabs",
        "boris_again",
        "gonzo_ML",
        "quant_prune_distill",
        "nadlskom",
        "elkornacio",
        "rybolos_channel",
        "def_model_train",
        "AIexTime",
        "new_yorko_times",
        "abstractDL",
        "partially_unsupervised",
        "epsiloncorrect",
        "mishin_learning",
        "korneychukov",
        "ruadaptnaya",
    ])
    records = spider()
    with open(output_file, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(download_channels)
