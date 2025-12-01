#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import fire  # type: ignore


class MessageThread(TypedDict):
    id: int
    text: str
    source: str
    pub_time: Optional[int]
    reply_to_message_id: Optional[int]
    replies: List["MessageThread"]
    urls: List[str]


def extract_text(text_data: Union[str, List[Union[str, Any]]]) -> str:
    if isinstance(text_data, str):
        return text_data

    text = ""
    for item in text_data:
        if isinstance(item, str):
            text += item
    return text


def build_thread_tree(
    root_msg: MessageThread, messages_by_parent: Dict[Tuple[str, int], List[MessageThread]]
) -> MessageThread:
    thread = root_msg
    msg_id = root_msg["id"]
    msg_source = root_msg["source"]
    thread["replies"] = []
    for reply in messages_by_parent[(msg_source, msg_id)]:
        reply_thread = build_thread_tree(reply, messages_by_parent)
        thread["replies"].append(reply_thread)
    return thread


def format_thread(thread: MessageThread, tab_level: int = 0) -> str:
    text = " ".join(thread["text"].split())
    text = f"{'--' * tab_level} {text}"
    for reply in thread["replies"]:
        text += f"\n{format_thread(reply, tab_level + 1)}"
    return text.strip()


def get_urls(thread: MessageThread) -> List[str]:
    assert thread["urls"]
    urls = thread["urls"][:]
    for reply in thread["replies"]:
        urls.extend(get_urls(reply))
    return urls


def extract_threads(
    input_file: Path,
    output_file: Path,
    min_text_length: int = 50,
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

    messages_by_parent: Dict[Tuple[str, int], List[MessageThread]] = defaultdict(list)
    root_messages: List[MessageThread] = list()

    print("Processing messages...")
    for msg in messages:
        if msg.get("type", "message") != "message":
            continue

        msg_id = msg.get("id")
        if not msg_id:
            continue

        text = extract_text(msg.get("text", ""))
        if not text:
            continue

        reply_to = msg.get("reply_to_message_id")

        url = msg.get("url")
        urls = []
        if url:
            urls = [url]

        source = msg.get("source")

        thread_msg = MessageThread(
            id=msg_id,
            source=source,
            pub_time=msg.get("pub_time"),
            text=text,
            reply_to_message_id=reply_to,
            replies=[],
            urls=urls,
        )

        if reply_to:
            messages_by_parent[(source, reply_to)].append(thread_msg)
        else:
            root_messages.append(thread_msg)

    complete_threads = []
    for root_msg in root_messages:
        thread_tree = build_thread_tree(root_msg, messages_by_parent)
        complete_threads.append(thread_tree)

    print(f"Found {len(complete_threads)} root threads")
    print(f"Writing results to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for thread in complete_threads:
            thread_text = format_thread(thread)
            if len(thread_text) < min_text_length:
                continue
            f.write(
                json.dumps(
                    {
                        "text": format_thread(thread),
                        "urls": get_urls(thread),
                        "source": thread["source"],
                        "pub_time": thread["pub_time"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Done!")


if __name__ == "__main__":
    fire.Fire(extract_threads)
