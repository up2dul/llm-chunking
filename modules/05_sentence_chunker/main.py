import os

from chonkie.chunker import SentenceChunker

chunker = SentenceChunker(
    chunk_size=200,  # Maximum number of tokens per chunk
)

with open("data.md", "r") as file:
    text = file.read()
    chunks = chunker(text)

chunked_context = ""
for chunk in chunks:
    chunked_context += chunk.text + "\n\n"
    chunked_context += "=" * 25 + "\n\n"

if not os.path.exists("results"):
    os.makedirs("results")
with open("results/sentece_chunked_token.md", "w") as file:
    file.write(chunked_context)
