from base.constants import Constants
from rag.bm25_keyword_retriever import BM25KeywordRetriever
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

import faiss
import numpy as np


class RAGCorpusManager:
    """Manage corpus ingestion, chunking, embedding, and indexing for RAG.

    Responsibilities:
        * Text normalization & deduplication to avoid redundant storage/embedding.
        * Sliding-window chunking with configurable length and stride.
        * Embedding computation via a provided `SentenceTransformer`.
        * FAISS (inner product) index creation/update for fast similarity search.

    Attributes:
        raw_dataset (list[tuple[str, dict]]): Original unique document entries as (text, metadata) pairs.
        chunked_data (list[str]): Generated text chunks.
        chunked_data_metadata (list[dict]): Metadata per chunk (document_id, data_chunk_id, start_word_index).
        sentence_embeddings (list[np.ndarray] or np.ndarray): Stored embeddings (if retained externally).
        sentence_transformer (SentenceTransformer): Model used for embedding generation.
        max_data_chunk_len (int): Maximum words per chunk.
        data_chunk_stride (int): Overlap stride between successive chunks.
    """

    def __init__(
        self,
        sentence_transformer: SentenceTransformer,
        max_data_chunk_len: int = 300,
        data_chunk_stride: int = 75,
    ):
        """Initialize the corpus manager.

        Args:
            sentence_transformer (SentenceTransformer): Preloaded sentence transformer used to encode text.
            max_data_chunk_len (int, optional): Maximum number of words per chunk. Defaults to 300.
            data_chunk_stride (int, optional): Overlap (in words) between consecutive chunks. Defaults to 75.

        Raises:
            ValueError: If `max_data_chunk_len` <= 0 or if stride is invalid (negative or >= max length).
        """
        self._document_id_tracker = 0
        self.raw_dataset = []
        self._unique_text_set = set()
        self.chunked_data = []
        self.chunked_data_metadata = []
        self.sentence_embeddings = []
        self.sentence_transformer = sentence_transformer
        self.max_data_chunk_len = max_data_chunk_len
        self.data_chunk_stride = data_chunk_stride
        self.bm25_retriever = BM25KeywordRetriever()
        if self.max_data_chunk_len <= 0:
            raise ValueError("Invalid chunk length; it should be a positive value")
        if self.data_chunk_stride < 0 or self.data_chunk_stride >= self.max_data_chunk_len:
            raise ValueError(
                "Invalid stride value; make sure that 0<= data_chunk_stride < max_data_chunk_len"
            )
        embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

    def clear_corpus(self) -> None:
        """
        Clears the existing corpus data and resets internal states.

        This method removes all stored documents, chunked data, embeddings,
        and resets the document ID tracker and unique text set. Should be used
        when starting a fresh corpus ingestion session only.
        """
        self._document_id_tracker = 0
        self.raw_dataset = []
        self._unique_text_set = set()
        self.chunked_data = []
        self.chunked_data_metadata = []
        self.sentence_embeddings = []
        self.bm25_retriever = BM25KeywordRetriever()
        embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalizes input text by trimming extra spaces and line breaks.

        Args:
            text (str): Input text string.

        Returns:
            str: Normalized text with single spaces between words.
        """
        return " ".join(text.strip().split())

    def _find_new_data_entries(
        self, texts: List[str], metadatas: List[dict]
    ) -> Tuple[List[str], List[dict]]:
        """Filter out duplicates from incoming texts.

        Normalizes text and checks against an internal set of seen strings.

        Args:
            texts (list[str]): Candidate text entries.
            metadatas (list[dict]): Corresponding metadata for each text entry.

        Returns:
            tuple[list[str], list[dict]]: Unique texts and their corresponding metadata
                whose normalized text was not previously stored.
        """
        new_unique_texts = []
        new_unique_metadatas = []
        for text, metadata in zip(texts, metadatas):
            normalized_text = self._normalize_text(text)
            if normalized_text and normalized_text not in self._unique_text_set:
                self._unique_text_set.add(normalized_text)
                new_unique_texts.append(normalized_text)
                new_unique_metadatas.append(metadata)
        return new_unique_texts, new_unique_metadatas

    def _chunk_data(self, texts: List[str], metadatas: List[dict]) -> Tuple[List[str], List[dict]]:
        """Chunk each document into overlapping word windows.

        Applies a sliding window of size `max_data_chunk_len` with stride
        `max_data_chunk_len - data_chunk_stride` until the end of the word list.
        Each chunk's metadata is the source document's metadata enriched with
        document_id, data_chunk_id, and start_word_index fields.

        Args:
            texts (list[str]): Text entries to chunk.
            metadatas (list[dict]): Corresponding metadata for each text entry.

        Returns:
            tuple[list[str], list[dict]]: Text chunks and their metadata dicts
                containing document_id, data_chunk_id, start_word_index, plus
                any fields from the original metadata.
        """
        data_chunks = []
        data_chunks_metadata = []
        for text, metadata in zip(texts, metadatas):
            if not text:
                continue
            words_list = text.split()
            if len(words_list) <= self.max_data_chunk_len:
                data_chunks.append(text)
                data_chunks_metadata.append(
                    {
                        **metadata,
                        Constants.CorpusManager.MetadataKeys.DOCUMENT_ID: self._document_id_tracker,
                        Constants.CorpusManager.MetadataKeys.DATA_CHUNK_ID: 0,
                        Constants.CorpusManager.MetadataKeys.START_WORD_INDEX: 0,
                    }
                )
                self._document_id_tracker += 1
                continue
            start_word_index = 0
            data_chunk_id = 0
            while start_word_index < len(words_list):
                chunk = words_list[start_word_index : start_word_index + self.max_data_chunk_len]
                data_chunks.append(" ".join(chunk))
                data_chunks_metadata.append(
                    {
                        **metadata,
                        Constants.CorpusManager.MetadataKeys.DOCUMENT_ID: self._document_id_tracker,
                        Constants.CorpusManager.MetadataKeys.DATA_CHUNK_ID: data_chunk_id,
                        Constants.CorpusManager.MetadataKeys.START_WORD_INDEX: start_word_index,
                    }
                )
                if start_word_index + self.max_data_chunk_len >= len(words_list):
                    break
                start_word_index += self.max_data_chunk_len - self.data_chunk_stride
                data_chunk_id += 1
            self._document_id_tracker += 1
        return data_chunks, data_chunks_metadata

    def _calculate_sentence_embeddings(
        self, data_chunks: List[str], batch_size: int = 64
    ) -> np.ndarray:
        """Compute embeddings for provided text chunks.

        Args:
            data_chunks (list[str]): Text chunks to encode.
            batch_size (int, optional): Batch size for model inference. Defaults to 64.

        Returns:
            np.ndarray: Embedding matrix shape (num_chunks, embedding_dim).
        """
        sentence_embeddings = self.sentence_transformer.encode(
            data_chunks, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return sentence_embeddings

    def _create_update_faiss_index(self, embeddings: np.ndarray):
        """Normalize embeddings and append them to the FAISS index.

        Embeddings are L2-normalized before insertion into the pre-initialized
        `IndexFlatIP` index.

        Args:
            embeddings (np.ndarray): Chunk embeddings to add.
        """
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def add_update_data_and_index(self, texts: List[str], metadatas: List[dict] = None):
        """Ingest new texts, embedding and indexing any unique content.

        Pipeline:
          1. Filter duplicates.
          2. Chunk new texts.
          3. Compute embeddings.
          4. Update FAISS index.
          5. Extend internal storage structures.

        Args:
            texts (list[str]): Incoming text entries to ingest.
            metadatas (list[dict], optional): Metadata dicts corresponding to each text entry.
                If None or not provided, empty dicts are used for each entry.

        Returns:
            None
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        new_texts, new_metadatas = self._find_new_data_entries(texts, metadatas)
        if not new_texts:
            return
        new_data_chunks, new_data_chunks_metadata = self._chunk_data(new_texts, new_metadatas)
        new_embeddings = self._calculate_sentence_embeddings(new_data_chunks)
        self._create_update_faiss_index(new_embeddings)
        self.raw_dataset.extend(list(zip(new_texts, new_metadatas)))
        self.chunked_data.extend(new_data_chunks)
        self.chunked_data_metadata.extend(new_data_chunks_metadata)
        self.bm25_retriever.build_index(self.chunked_data)

    def mark_chunks_with_same_document_id_as_deleted(self, index: int) -> List[int]:
        """Find all chunks and metadata associated with the same document_id.

        This method removes entries from `chunked_data`, `chunked_data_metadata`,
        and the FAISS index corresponding to the specified `document_id`. It does
        not modify the original `raw_dataset` or the unique text set, as those
        are only used for ingestion and deduplication.

        Args:
            index (int): The index of a chunk whose document ID should be used to find all related chunks.
        """
        document_id = self.chunked_data_metadata[index][
            Constants.CorpusManager.MetadataKeys.DOCUMENT_ID
        ]
        for i in range(index - 1, -1, -1):
            if (
                self.chunked_data_metadata[i][Constants.CorpusManager.MetadataKeys.DOCUMENT_ID]
                == document_id
            ):
                self.chunked_data_metadata[i][Constants.CorpusManager.MetadataKeys.DELETED] = True
            else:
                break
        for i in range(index + 1, len(self.chunked_data_metadata)):
            if (
                self.chunked_data_metadata[i][Constants.CorpusManager.MetadataKeys.DOCUMENT_ID]
                == document_id
            ):
                self.chunked_data_metadata[i][Constants.CorpusManager.MetadataKeys.DELETED] = True
            else:
                break
