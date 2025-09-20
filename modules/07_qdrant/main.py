from chonkie.chunker import RecursiveChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Basic Qdrant vector database setup for semantic search
# This implementation demonstrates single dense vector storage and retrieval
collection_name = "first-collection"

# SentenceTransformers: production-ready, multilingual embedding model
# all-MiniLM-L6-v2: balanced performance vs speed (384 dimensions)
# Excellent for general-purpose semantic similarity tasks
# Reference: https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant client for local development
qdrant = QdrantClient(url="http://localhost:6333")

# Clean slate approach: recreate collection for consistent schema
# In production, consider update operations instead of full recreation
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name=collection_name)

# Create vector collection optimized for semantic search
# Vector size must match embedding model output dimensions
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,  # all-MiniLM-L6-v2 embedding dimensions
        distance=Distance.COSINE,  # Ideal for normalized embeddings
    ),
)

# RecursiveChunker: intelligent text segmentation preserving semantic boundaries
# Splits hierarchically: paragraphs → sentences → tokens for optimal context
# Better retrieval accuracy than fixed-size chunking
chunker = RecursiveChunker(
    chunk_size=2000,  # Optimal balance: context vs embedding model limits
)

# Load and process document text for vector database storage
# Chunking enables handling of large documents exceeding model context limits
with open("data.md", "r") as file:
    text = file.read()
    # RecursiveChunker maintains semantic coherence in each chunk
    chunks = chunker(text)

# Batch processing: collect all vectors before upload for efficiency
# Reduces network overhead compared to individual point uploads
vector_data = []

# Generate embeddings for each chunk and prepare for vector storage
# Sequential IDs provide simple ordering and debugging capabilities
for i, chunk in enumerate(chunks, start=1):
    chunked_text = chunk.text

    # SentenceTransformers.encode: converts text to 384-dimensional vector
    # Captures semantic meaning for similarity-based retrieval
    # Model automatically handles tokenization and normalization
    embeddings = model.encode(chunked_text)

    # PointStruct: Qdrant's data structure for vector + metadata storage
    # Payload stores original text for result display and post-processing
    vector_data.append(
        PointStruct(
            id=i,  # Sequential ID for simple chunk identification
            vector=embeddings,  # Dense semantic representation
            payload={"text": chunked_text},  # Original content for retrieval
        )
    )

# Batch upload: efficient bulk insertion into vector database
# Single operation reduces connection overhead and improves performance
qdrant.upload_points(
    collection_name=collection_name,
    points=vector_data,
)
