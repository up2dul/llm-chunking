from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(
    api_key=getenv("OPENAI_API_KEY"),
    base_url=getenv("OPENAI_BASE_URL"),
)


def main() -> None:
    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
