from rag.rag_corpus_manager import RAGCorpusManager
from typing import List, Tuple

import faiss
import heapq
import torch


class RAGContentRetriever:
    """
    Handle end-to-end passage retrieval in a Retrieval-Augmented Generation (RAG) system.

    Integrates two main components:
      1. Bi-encoder (FAISS index) for fast passage retrieval.
      2. Cross-encoder for semantic re-ranking of candidate passages.
    """

    def __init__(
        self,
        cross_encoder,
        rrf_k: int = 60,
    ):
        """
        Initialize the RAGContentRetriever.

        Args:
            cross_encoder: Pretrained cross-encoder model (e.g., from sentence-transformers).
            rrf_k (int): Rank-smoothing constant for Reciprocal Rank Fusion. Defaults to 60.

        RRF score for a document d:
            rrf(d) = sum_r  1 / (k + rank_r(d))

        where rank_r(d) is the 1-based rank of d in retriever result list r.
        Documents absent from a result list are simply not counted.
        """
        self._cross_encoder = cross_encoder
        self._rrf_k = rrf_k

    def _merge_passage_chunks_and_scores(self, chunks_list) -> Tuple[str, float]:
        """
        Merge overlapping text chunks and compute their average relevance score.

        Args:
            chunks_list (list[tuple]): List of tuples, each containing:
                - start_word_index (int): Start position of the chunk in the original text.
                - chunk_text (str): Chunk text content.
                - score (float): Relevance score assigned by the cross-encoder.

        Returns:
            tuple:
                str: Merged text string from the chunks.
                float: Average relevance score across the merged chunks.
        """
        merged_text_list = []
        chunk_score = 0
        for i in range(len(chunks_list)):
            if (
                i < len(chunks_list) - 1
                and chunks_list[i + 1][0] <= chunks_list[i][0] + len(chunks_list[i][1]) - 1
            ):
                merged_text_list.append(
                    chunks_list[i][1][: chunks_list[i + 1][0] - chunks_list[i][0]]
                )
            else:
                merged_text_list.append(chunks_list[i][1])
            chunk_score += chunks_list[i][2]
        return " ".join(merged_text_list), chunk_score / len(chunks_list)

    def _biencoder_find_top_similar_items(
        self, query: str, top_k: int, rag_corpus: RAGCorpusManager
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k semantically similar passages using the FAISS bi-encoder index.

        Args:
            query (str): Query text to search for.
            top_k (int): Number of top results to retrieve.
            rag_corpus (RAGCorpusManager): Corpus manager containing FAISS index and embeddings.

        Returns:
            tuple[list[int], list[float]]: Indices and corresponding similarity scores of the top-k chunks.
        """
        query_embedding = rag_corpus.sentence_transformer.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = rag_corpus.faiss_index.search(query_embedding, top_k)
        return indices[0].tolist(), scores[0].tolist()

    def _hybrid_score_and_rank(
        self,
        biencoder_indices: List[int],
        bm25_indices: List[int],
        semantic_weight: float,
        lexical_weight: float,
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """
        Combine bi-encoder and BM25 ranked lists via Reciprocal Rank Fusion (RRF).

        RRF score for a document d across result lists r:
            rrf(d) = sum_r  w_r / (k + rank_r(d))

        where rank_r(d) is the 1-based position of d in list r, w_r is the
        corresponding retriever weight (semantic_weight / lexical_weight), and
        k is the rank-smoothing constant (default 60). Documents absent from a
        list are simply ignored for that list's contribution.

        Score range (with default k=60):
            - Single retriever, rank 1:  w / (60 + 1) ≈ 0.0164 * w
            - Both retrievers, both rank 1:  (w_s + w_l) / 61

        Args:
            biencoder_indices (list[int]): Chunk indices returned by the bi-encoder, ordered by descending score.
            bm25_indices (list[int]): Chunk indices returned by BM25, ordered by descending score.
            semantic_weight (float): Weight applied to the bi-encoder ranked list contribution.
            lexical_weight (float): Weight applied to the BM25 ranked list contribution.
            top_k (int): Number of top-ranked indices to return.

        Returns:
            tuple[list[int], list[float]]: Indices and corresponding RRF scores of the top-k chunks.
        """
        k = self._rrf_k
        rrf_scores: dict = {}
        for rank, idx in enumerate(biencoder_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + semantic_weight / (k + rank)
        for rank, idx in enumerate(bm25_indices, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + lexical_weight / (k + rank)

        top_k_indices = heapq.nlargest(top_k, rrf_scores, key=rrf_scores.__getitem__)
        top_k_scores = [rrf_scores[idx] for idx in top_k_indices]
        return top_k_indices, top_k_scores

    def _filter_indices_by_metadata(
        self,
        rag_corpus: RAGCorpusManager,
        indices: List[int],
        filter_out_items_using_metadata: dict,
    ) -> List[int]:
        """
        Filter out passage chunk indices whose metadata matches ALL key-value pairs
        in ``filter_out_items_using_metadata``.

        Args:
            rag_corpus (RAGCorpusManager): Corpus manager containing chunked data metadata.
            indices (list[int]): Candidate chunk indices to filter.
            filter_out_items_using_metadata (dict): Metadata key-value pairs; any chunk
                whose metadata matches **all** pairs is excluded.

        Returns:
            list[int]: Indices with matching chunks removed.
        """
        filtered = []
        for idx in indices:
            metadata = rag_corpus.chunked_data_metadata[idx]
            if not all(
                metadata.get(key, None) == value
                for key, value in filter_out_items_using_metadata.items()
            ):
                filtered.append(idx)
        return filtered

    def _cross_encoder_find_top_similar_items(
        self,
        query: str,
        rag_corpus: RAGCorpusManager,
        passage_chunk_indices: List[int],
        top_k: int = 5,
        batch_size: int = 64,
    ) -> Tuple[List[int], List[float]]:
        """
        Re-rank retrieved passages using a cross-encoder for more precise semantic relevance.

        Metadata filtering is expected to be applied to ``passage_chunk_indices`` before
        calling this method (see ``find_top_similar_items``).

        Args:
            query (str): Input query text.
            rag_corpus (RAGCorpusManager): Corpus manager containing chunked data and metadata.
            passage_chunk_indices (list[int]): Indices of passage chunks to re-rank.
            top_k (int, optional): Number of top passages to keep after re-ranking. Defaults to 5.
            batch_size (int, optional): Number of passage pairs to process in a single batch. Defaults to 64.

        Returns:
            tuple[list[int], list[float]]: Indices and corresponding cross-encoder scores of the
            top-k chunks, sorted by descending relevance.
        """
        self._cross_encoder.eval()
        scores = []

        with torch.no_grad():
            for start in range(0, len(passage_chunk_indices), batch_size):
                batch_indices = passage_chunk_indices[start : start + batch_size]
                batch_passages = [rag_corpus.chunked_data[i] for i in batch_indices]
                batch_pairs = [[query, passage] for passage in batch_passages]
                batch_scores = self._cross_encoder.predict(batch_pairs)
                scores.extend(zip(batch_indices, batch_scores))

        most_related_items_index_score_pairs = sorted(scores, key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in most_related_items_index_score_pairs[:top_k]]
        top_scores = [float(score) for _, score in most_related_items_index_score_pairs[:top_k]]
        return top_indices, top_scores

    def find_top_similar_items(
        self,
        rag_corpus: RAGCorpusManager,
        query: str,
        initial_retrieval_top_k: int = 20,
        top_k: int = 5,
        cross_encoder_batch_size: int = 16,
        semantic_similarity_weight: float = 1.0,
        lexical_similarity_weight: float = 0.0,
        filter_out_items_using_metadata: dict = None,
        use_cross_encoder: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top similar passages to the query using hybrid retrieval.

        Bi-encoder and BM25 ranked lists are fused with Reciprocal Rank Fusion
        (RRF, k=60) before optionally being re-ranked by the cross-encoder.

        Args:
            rag_corpus (RAGCorpusManager): Corpus manager containing documents, embeddings, and metadata.
            query (str): User query string.
            initial_retrieval_top_k (int, optional): Number of passages retrieved by each of the bi-encoder and BM25. Defaults to 20.
            top_k (int, optional): Number of passages kept after re-ranking. Defaults to 5.
            cross_encoder_batch_size (int, optional): Batch size for cross-encoder inference. Defaults to 16.
            semantic_similarity_weight (float, optional): Weight for semantic (bi-encoder) score. Defaults to 1.0.
            lexical_similarity_weight (float, optional): Weight for lexical (BM25) score. Defaults to 0.0.
            filter_out_items_using_metadata (dict, optional): Metadata key-value pairs to filter out passages. Defaults to None.
            use_cross_encoder (bool, optional): Whether to re-rank candidates with the cross-encoder. Defaults to True.

        Returns:
            tuple[list[int], list[float]]: Indices and corresponding scores of the
            top-k chunks, sorted by descending relevance.
        """
        if rag_corpus.faiss_index.ntotal == 0:
            return [], []
        if semantic_similarity_weight == 0.0:
            biencoder_indices = []
        else:
            biencoder_indices, _ = self._biencoder_find_top_similar_items(
                query, initial_retrieval_top_k, rag_corpus
            )
        if lexical_similarity_weight == 0.0:
            bm25_indices = []
        else:
            bm25_indices, _ = rag_corpus.bm25_retriever.search(query, top_k=initial_retrieval_top_k)
        hybrid_top_indices, hybrid_top_scores = self._hybrid_score_and_rank(
            biencoder_indices,
            bm25_indices,
            semantic_similarity_weight,
            lexical_similarity_weight,
            initial_retrieval_top_k,
        )

        if filter_out_items_using_metadata is not None:
            hybrid_top_indices = self._filter_indices_by_metadata(
                rag_corpus, hybrid_top_indices, filter_out_items_using_metadata
            )
            # Re-align scores to the filtered indices
            index_to_score = dict(zip(hybrid_top_indices, hybrid_top_scores))
            hybrid_top_scores = [index_to_score[idx] for idx in hybrid_top_indices]

        if not use_cross_encoder:
            return hybrid_top_indices[:top_k], hybrid_top_scores[:top_k]

        cross_encoder_top_indices, cross_encoder_top_scores = (
            self._cross_encoder_find_top_similar_items(
                query,
                rag_corpus,
                hybrid_top_indices,
                top_k,
                cross_encoder_batch_size,
            )
        )
        return cross_encoder_top_indices, cross_encoder_top_scores
