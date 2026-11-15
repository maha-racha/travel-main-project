from slowapi import Limiter
from slowapi.util import get_remote_address
import time
import asyncio

limiter = Limiter(key_func=get_remote_address)

class TokenBucket:
    def __init__(self, rate: int, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Refill tokens
            new_tokens = elapsed * (self.rate / 60.0) # rate is per minute
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
