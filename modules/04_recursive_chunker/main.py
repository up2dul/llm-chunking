import os

from chonkie.chunker import RecursiveChunker

chunker = RecursiveChunker(
    chunk_size=2000,  # Maximum number of tokens per chunk
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
with open("results/recursive_chunked_token.md", "w") as file:
    file.write(chunked_context)
