from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedSparseVector,
    NamedVector,
    SearchRequest,
    SparseVector,
)
from utils import rrf_fusion

load_dotenv()

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

# User query for hybrid search
question_query = "Who is the founder of Tokopedia?"

# Generate dense embedding for semantic similarity
dense_res = openai.embeddings.create(
    input=question_query,
    model="text-embedding-3-small",
)
dense_query_vector = dense_res.data[0].embedding

# Generate sparse embedding for keyword matching
sparse_embeddings = list(sparse_model.embed([question_query]))[0]
sparse_query_vector = SparseVector(
    indices=sparse_embeddings.indices.tolist(),
    values=sparse_embeddings.values.tolist(),
)

# Perform hybrid search: semantic + keyword matching in parallel
search_results = qdrant.search_batch(
    collection_name=collection_name,
    requests=[
        SearchRequest(
            vector=NamedVector(
                name=dense_vector_name,
                vector=dense_query_vector,
            ),
            limit=10,
            with_payload=True,
        ),
        SearchRequest(
            vector=NamedSparseVector(
                name=sparse_vector_name,
                vector=sparse_query_vector,
            ),
            limit=10,
            with_payload=True,
        ),
    ],
)

# Combine results using Reciprocal Rank Fusion for better relevance
fusion_results = rrf_fusion(search_results[0], search_results[1])
cleaned_documents = [
    {
        "id": point.id,
        "score": point.score,
        "payload": point.payload,
    }
    for point in fusion_results
]

# Generate answer using retrieved context (RAG pattern)
res = openai.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": f""""
                You are a helpful assistant.
                You will answer questions based on the given context.
                
                Context:
                {cleaned_documents}
            """,
        },
        {"role": "user", "content": question_query},
    ],
)

print(res.choices[0].message.content)
