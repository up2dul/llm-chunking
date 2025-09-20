from chonkie.chunker import RecursiveChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

collection_name = "first-collection"
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")

if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name=collection_name)
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,  # It depends on the model that used
        distance=Distance.COSINE,
    ),
)

chunker = RecursiveChunker(
    chunk_size=2000,  # Maximum number of tokens per chunk
)

with open("data.md", "r") as file:
    text = file.read()
    chunks = chunker(text)

vector_data = []

for i, chunk in enumerate(chunks, start=1):
    chunked_text = chunk.text
    embeddings = model.encode(chunked_text)
    vector_data.append(
        PointStruct(
            id=i,
            vector=embeddings,
            payload={"text": chunked_text},
        )
    )

qdrant.upload_points(
    collection_name=collection_name,
    points=vector_data,
)
