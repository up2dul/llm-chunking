import os

from chonkie.chunker import SemanticChunker

chunker = SemanticChunker(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_sentences=10,
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
with open("results/semantic_chunked_token.md", "w") as file:
    file.write(chunked_context)
