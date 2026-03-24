from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class BM25KeywordRetriever:
    def __init__(self, tokenizer=word_tokenize):
        self.bm_25_index = None
        self.word_tokenizer = tokenizer
        self.stemmer = PorterStemmer()

    def _tokenize_and_stem(self, text: str) -> list[str]:
        """Lowercase, tokenize, and stem text to base word forms.

        Args:
            text (str): Input text to process.
        Returns:
            list[str]: List of stemmed tokens.
        """
        tokens = self.word_tokenizer(text.lower())
        return [self.stemmer.stem(token) for token in tokens]

    def build_index(self, documents: list[str]):
        """Build the BM25 index from a list of documents.

        Args:
            documents (list[str]): List of documents to index.
        """
        tokenized_documents = [self._tokenize_and_stem(doc) for doc in documents]
        self.bm_25_index = BM25Okapi(tokenized_documents)

    def search(self, query: str, top_k: int) -> tuple[list[int], list[float]]:
        """Retrieve top-k relevant documents based on BM25 keyword matching

        Args:
            query (str): User query string.
            top_k (int): Number of top relevant documents to retrieve.
        Returns:
            tuple[list[int], list[float]]: Indices and corresponding BM25 scores of the
            top-k most relevant documents, both ordered by descending score.
        """
        if self.bm_25_index is None:
            raise ValueError("BM25 index has not been built. Call build_index() first.")
        tokenized_query = self._tokenize_and_stem(query)
        doc_scores = self.bm_25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[
            :top_k
        ]
        top_scores = [doc_scores[i] for i in top_indices]
        return top_indices, top_scores
