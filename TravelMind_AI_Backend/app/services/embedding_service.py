import openai
import logging
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
from app.core.config import Settings
from rank_bm25 import BM25Okapi

from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            max_retries=3,
            timeout=60.0
        )
        self.model = settings.openai_embedding_model

    async def get_embedding(self, text: str) -> List[float]:
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                if hasattr(response, 'usage'):
                    logger.info(f"Embedding Token Usage: Input={response.usage.prompt_tokens}, Total={response.usage.total_tokens}")
                return response.data[0].embedding
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating embedding after {max_retries} retries: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                if hasattr(response, 'usage'):
                    logger.info(f"Batch Embedding Token Usage: Input={response.usage.prompt_tokens}, Total={response.usage.total_tokens}")
                return [item.embedding for item in response.data]
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating batch embeddings after {max_retries} retries: {str(e)}")
                    raise
            except openai.APIError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"API error, waiting {wait_time}s before retry: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating batch embeddings after {max_retries} retries: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise

    def get_embedding_dimension(self) -> int:
        if "large" in self.model:
            return 3072
        elif "small" in self.model:
            return 1536
        else:
            return 1536

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25:
        1. Lowercase
        2. Replace diacritics (ă->a, î->i, etc.)
        3. Remove punctuation/special chars
        4. Split by whitespace
        """
        import unicodedata
        import re
        
        if not text:
            return []
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Normalize diacritics (NFD decomposition -> strip combing chars)
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        
        # 3. Replace non-alphanumeric with space
        # This keeps numbers but removes punctuation
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 4. Split and filter empty
        tokens = [t for t in text.split() if len(t) > 1] # Remove single chars?
        
        return tokens

    def _load_bm25_params(self):
        """Load pre-fitted BM25 parameters from pickle file"""
        if hasattr(self, "_bm25_params"):
            return
        
        try:
            model_path = Path(__file__).parent / "models" / "bm25_params.pkl"
            
            if not model_path.exists():
                logger.warning(f"BM25 model not found at {model_path}. Run scripts/fit_bm25_model.py first.")
                logger.warning("Falling back to simple term frequency (no IDF weighting).")
                self._bm25_params = None
                return
            
            with open(model_path, 'rb') as f:
                params = pickle.load(f)
            
            # Validate loaded params
            required_keys = ['doc_freqs', 'idf', 'avgdl', 'corpus_size']
            if not all(key in params for key in required_keys):
                logger.error(f"Invalid BM25 params file. Missing required keys. Re-run fit_bm25_model.py")
                self._bm25_params = None
                return
            
            self._bm25_params = params
            logger.info(f"Loaded BM25 model with {len(self._bm25_params['doc_freqs'])} unique terms")
        except Exception as e:
            logger.error(f"Error loading BM25 parameters: {str(e)}")
            self._bm25_params = None

    def get_sparse_embedding(self, text: str) -> Dict[str, List]:
        """
        Generate sparse vector using BM25 algorithm.
        Returns a dictionary suitable for Qdrant sparse vector format:
        {'indices': [1, 20, ...], 'values': [0.5, 0.3, ...]}
        """
        # Lazy load BM25 parameters
        if not hasattr(self, "_bm25_params"):
            self._load_bm25_params()
        
        try:
            # Tokenize the query
            tokens = self._tokenize(text)
            
            if not tokens:
                return {"indices": [], "values": []}
            
            # If BM25 params not available, return simple bag-of-words
            if self._bm25_params is None:
                # Simple term frequency without IDF
                term_freq = {}
                for token in tokens:
                    term_freq[token] = term_freq.get(token, 0) + 1
                
                # Create vocabulary mapping
                vocab = {term: idx for idx, term in enumerate(sorted(term_freq.keys()))}
                indices = [vocab[term] for term in term_freq.keys()]
                values = [float(freq) for freq in term_freq.values()]
                
                return {"indices": indices, "values": values}
            
            # Use BM25 scoring with pre-fitted parameters
            doc_freqs = self._bm25_params['doc_freqs']
            idf = self._bm25_params['idf']
            avgdl = self._bm25_params['avgdl']
            corpus_size = self._bm25_params['corpus_size']
            
            # BM25 parameters
            k1 = 1.5
            b = 0.75
            
            # Calculate term frequencies for query
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            doc_len = len(tokens)
            
            # Calculate BM25 score for each term
            scores = {}
            for term, freq in term_freq.items():
                if term in idf:
                    # BM25 formula
                    term_idf = idf[term]
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * (doc_len / avgdl))
                    score = term_idf * (numerator / denominator)
                    scores[term] = score
            
            if not scores:
                return {"indices": [], "values": []}
            
            # Create vocabulary mapping for terms in this query
            vocab = {term: idx for idx, term in enumerate(sorted(scores.keys()))}
            
            # Convert to Qdrant sparse vector format
            indices = [vocab[term] for term in scores.keys()]
            values = [float(scores[term]) for term in scores.keys()]
            
            return {"indices": indices, "values": values}
            
        except Exception as e:
            logger.error(f"Error generating BM25 sparse embedding: {str(e)}")
            # Return empty sparse vector on failure to avoid breaking sync
            return {"indices": [], "values": []}

