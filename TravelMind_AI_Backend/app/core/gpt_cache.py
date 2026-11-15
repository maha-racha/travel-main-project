import logging
import os
from gptcache import cache
from gptcache.manager import get_data_manager
# from langchain.globals import set_llm_cache # Fails in some environments
from langchain_core.globals import set_llm_cache

from app.core.config import get_settings

logger = logging.getLogger(__name__)

from typing import Optional, Sequence
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

class GPTCacheAdapter(BaseCache):
    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        """Look up based on prompt."""
        # search returns List[Tuple[question, answer, embedding, tags]]
        # We assume exact match or map based search returns non-empty list if found.
        if not cache.data_manager:
            return None
            
        res = cache.data_manager.search(prompt)
        if res and len(res) > 0:
            # Return the answer (second element in tuple)
            return res[0][1]
        return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        """Update cache."""
        if cache.data_manager:
            cache.data_manager.save(prompt, return_val, prompt)

    def clear(self, **kwargs) -> None:
        """Clear cache."""
        # GPTCache currently doesn't expose a simple clear method on the global object easily
        # or it depends on the data manager.
        # We can implement this if needed, but for now we pass.
        pass

def init_gptcache(embedding_function=None):
    """
    Initialize GPTCache and set it as the global LLM cache for LangChain.
    
    Args:
        embedding_function: Optional function to generate embeddings. 
                            If provided, semantic caching will be enabled.
                            If None, exact match caching will be used.
    """
    logger.info("Initializing GPTCache...")
    
    # Define a common cache file path
    cache_dir = "gptcache_data"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # We will use a simple map-based cache for now (Exact Match)
    # This is the most reliable start without adding heavy vector DB dependencies for the cache itself
    # independent of the main app's vector store.
    
    # If we want to use the main app's embedding service for semantic cache, we can pass it here.
    # For now, let's start with Exact Match to ensure stability.
    
    cache_file = os.path.join(cache_dir, "cache_map.pkl")
    data_manager = get_data_manager(data_path=cache_file)
    
    cache.init(
        pre_embedding_func=get_content_func, # Function to extract content from prompt
        data_manager=data_manager
    )
    
    set_llm_cache(GPTCacheAdapter())
    logger.info("GPTCache initialized.")

def get_content_func(data, **kwargs):
    """
    Custom function to extract text content from the LangChain prompt input.
    LangChain often sends a string or a list of messages.
    """
    if isinstance(data, str):
        return data
    if hasattr(data, "content"):
        return data.content
    # If it's a list of messages (which ChatOpenAI uses)
    if isinstance(data, list):
        return "\n".join([msg.content for msg in data if hasattr(msg, "content")])
        
    return str(data)
