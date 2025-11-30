#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union, Any
from collections import defaultdict

import fire  # type: ignore


class MessageThread(TypedDict):
    id: int
    text: str
    reply_to_message_id: Optional[int]
    replies: List["MessageThread"]


def extract_text(text_data: Union[str, List[Union[str, Any]]]) -> str:
    if isinstance(text_data, str):
        return text_data

    text = ""
    for item in text_data:
        if isinstance(item, str):
            text += item
    return text


def build_thread_tree(
    root_msg: MessageThread, messages_by_parent: Dict[int, List[MessageThread]]
) -> MessageThread:
    thread = root_msg
    msg_id = root_msg["id"]
    thread["replies"] = []
    for reply in messages_by_parent[msg_id]:
        reply_thread = build_thread_tree(reply, messages_by_parent)
        thread["replies"].append(reply_thread)
    return thread


def format_thread(thread: MessageThread, tab_level: int = 0) -> str:
    text = " ".join(thread["text"].split())
    text = f"{'--' * tab_level} {text}"
    for reply in thread["replies"]:
        text += f"\n{format_thread(reply, tab_level + 1)}"
    return text


def get_urls(thread: MessageThread, root_url: str) -> List[str]:
    urls = [root_url + str(thread["id"])]
    for reply in thread["replies"]:
        urls.extend(get_urls(reply, root_url))
    return urls


def process_dump(
    input_file: Path, output_file: Path, root_url: str = "https://t.me/natural_language_processing/"
) -> None:
    print(f"Reading {input_file}...")

    if str(input_file).endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])
    else:
        assert str(input_file).endswith(".jsonl")
        with open(input_file, "r", encoding="utf-8") as f:
            messages = [json.loads(line) for line in f]

    messages_by_parent: Dict[int, List[MessageThread]] = defaultdict(list)
    root_messages: List[MessageThread] = list()

    print("Processing messages...")
    for msg in messages:
        if msg.get("type") != "message":
            continue

        msg_id = msg.get("id")
        if not msg_id:
            continue

        text = extract_text(msg.get("text", ""))
        if not text:
            continue

        reply_to = msg.get("reply_to_message_id")

        thread_msg = MessageThread(id=msg_id, text=text, reply_to_message_id=reply_to, replies=[])

        if reply_to:
            messages_by_parent[reply_to].append(thread_msg)
        else:
            root_messages.append(thread_msg)

    complete_threads = []
    for root_msg in root_messages:
        thread_tree = build_thread_tree(root_msg, messages_by_parent)
        complete_threads.append(thread_tree)

    print(f"Found {len(complete_threads)} root threads")
    print(f"Writing results to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threads": [
                    {
                        "text": format_thread(thread),
                        "urls": get_urls(thread, root_url=root_url),
                    }
                    for thread in complete_threads
                ]
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done!")


if __name__ == "__main__":
    fire.Fire(process_dump)
