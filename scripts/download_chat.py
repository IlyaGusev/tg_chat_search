import os
import json

import fire  # type: ignore
from tqdm.asyncio import tqdm  # type: ignore
from dotenv import load_dotenv
from telethon import TelegramClient  # type: ignore


CHATS = {
    "natural_language_processing": -1001095835958
}
URL_TEMPLATE = "https://t.me/{}/{}"


async def download_chats(output_path: str) -> None:
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")

    client = TelegramClient("get_chat", api_id, api_hash)
    await client.start()

    with open(output_path, "w") as f:
        for chat_name, chat_id in CHATS.items():
            chat = await client.get_input_entity(chat_id)
            async for message in tqdm(client.iter_messages(chat)):
                message_id = message.id
                if not message.message:
                    continue
                text = message.message.strip()
                if not text:
                    continue
                reply_to_message_id = message.reply_to.reply_to_msg_id if message.reply_to else None
                record = {
                    "id": message_id,
                    "url": URL_TEMPLATE.format(chat_name, message_id),
                    "type": "message",
                    "text": text,
                    "reply_to_message_id": reply_to_message_id,
                    "chat_id": chat_id,
                    "pub_time": int(message.date.timestamp()),
                    "source": chat_name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(download_chats)
