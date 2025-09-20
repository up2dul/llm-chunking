import uuid

from chonkie.chunker import RecursiveChunker
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

# Initialize Qdrant client for local development
qdrant = QdrantClient(
    url="http://localhost:6333",
)
openai = OpenAI()

# Configuration for multivector collection supporting both dense and sparse embeddings
# This hybrid approach combines semantic similarity (dense) with exact term matching (sparse)
collection_name = "multivector-collection"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
dense_vector_size = 1536  # OpenAI text-embedding-3-small dimensions
# SPLADE model for sparse embeddings - excels at keyword/term matching
# Reference: https://github.com/naver/splade
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# RecursiveChunker breaks text hierarchically: paragraphs → sentences → tokens
# Preserves semantic boundaries better than fixed-size chunking
# Docs: https://github.com/chonkie-inc/chonkie
chunker = RecursiveChunker(
    chunk_size=2000,  # Maximum tokens per chunk for optimal context window usage
)


# Clean slate: recreate collection to ensure consistent schema
# In production, consider using update operations instead of recreation
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name=collection_name)

# Create multivector collection supporting hybrid search
# Dense vectors: capture semantic meaning and context
# Sparse vectors: capture exact term matches and rare keywords
# Reference: https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config={
        dense_vector_name: VectorParams(
            size=dense_vector_size,
            distance=Distance.COSINE,  # Cosine similarity for normalized embeddings
        ),
    },
    sparse_vectors_config={
        sparse_vector_name: SparseVectorParams(),  # Dynamic dimensionality
    },
)

# Load and process the document text
# Using chunking to handle large documents that exceed embedding model limits
with open("data.md", "r") as file:
    text = file.read()
    # RecursiveChunker splits intelligently at natural boundaries
    # Better retrieval accuracy than arbitrary cuts mid-sentence
    chunks = chunker(text)

# Process each chunk with dual embedding strategy
# This hybrid approach maximizes both semantic and lexical recall
for chunk in chunks:
    chunked_context = chunk.text

    # Dense embeddings: capture semantic similarity and context
    # OpenAI's text-embedding-3-small: fast, high-quality, cost-effective
    # Ideal for semantic search and finding conceptually similar content
    dense_embeddings = (
        openai.embeddings.create(
            input=chunked_context,
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )

    # Sparse embeddings: capture exact term matches and importance weighting
    # SPLADE model learns which terms are most important for retrieval
    # Excels at finding documents with specific keywords or rare terms
    sparse_embeddings = list(sparse_model.embed([chunked_context]))[0]
    # Convert to Qdrant's SparseVector format for efficient storage
    sparse_vector = SparseVector(
        indices=sparse_embeddings.indices.tolist(),
        values=sparse_embeddings.values.tolist(),
    )

    # Store both vector types with the same point for unified hybrid search
    # Each document chunk becomes searchable via both semantic and lexical queries
    qdrant.upload_points(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),  # Unique identifier for each chunk
                vector={
                    dense_vector_name: dense_embeddings,
                    sparse_vector_name: sparse_vector,
                },
                # Store original text for result display and further processing
                payload={"text": chunked_context, "document_id": str(uuid.uuid4())},
            ),
        ],
    )
