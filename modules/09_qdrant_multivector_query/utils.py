from typing import Any


# Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists into a single, better ranking.
# It's widely used in search engines and recommendation systems
# RRF_score(item) = Sigma(1 / (k + rank_i))
def rrf_fusion(dense_results, sparse_results, alpha=60) -> list[Any]:
    """
    Rank-based fusion of dense and sparse results.
    Combines the scores of both types of results using RRF.
    60-70 is a sweet spot for alpha, which controls the importance of dense results.
    Args:
        dense_results (list): List of dense results.
        sparse_results (list): List of sparse results.
        alpha (int): Weighting factor for sparse results.
    Returns:
        list: List of combined results.
    """
    scores = {}

    for rank, point in enumerate(dense_results, 1):
        scores[point.id] = scores.get(point.id, 0) + 1.0 / (alpha + rank)

    for rank, point in enumerate(sparse_results, 1):
        scores[point.id] = scores.get(point.id, 0) + 1.0 / (alpha + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    result_points = []
    all_points = {p.id: p for p in dense_results + sparse_results}

    for point_id in sorted_ids[:10]:
        point = all_points[point_id]
        point.score = scores[point_id]
        result_points.append(point)

    return result_points
