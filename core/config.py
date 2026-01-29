from dotenv import load_dotenv
import os

def load_env() -> None:
    load_dotenv()

def get_news_api_key() -> str | None:
    return os.getenv("NEWS_API_KEY")
