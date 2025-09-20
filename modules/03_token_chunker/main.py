import os

from chonkie.chunker import TokenChunker

chunker = TokenChunker(
    chunk_size=2000,  # Maximum number of tokens per chunk
    chunk_overlap=300,  # Number of tokens to overlap between chunks
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
with open("results/chunked_token.md", "w") as file:
    file.write(chunked_context)
