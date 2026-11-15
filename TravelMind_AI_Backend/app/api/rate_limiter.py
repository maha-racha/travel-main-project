import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleRateLimiter:
    """
    Simple in-memory rate limiter to prevent DDoS attacks.
    
    Limits: 10 requests per minute per user/IP
    Auto-cleanup: Removes old entries every 5 minutes
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Store: {identifier: [(timestamp1, timestamp2, ...)]}
        self.requests: Dict[str, list] = defaultdict(list)
        
        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if request is allowed.
        
        Args:
            identifier: User ID or IP address
            
        Returns:
            (is_allowed, requests_remaining)
        """
        current_time = time.time()
        
        # Periodic cleanup to prevent memory bloat
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
        
        # Get request history for this identifier
        request_times = self.requests[identifier]
        
        # Remove timestamps outside the current window
        cutoff_time = current_time - self.window_seconds
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if limit exceeded
        if len(request_times) >= self.max_requests:
            remaining = 0
            allowed = False
            logger.warning(
                f"[RATE_LIMIT] User {identifier} exceeded limit "
                f"({len(request_times)}/{self.max_requests} requests)"
            )
        else:
            # Add current request
            request_times.append(current_time)
            remaining = self.max_requests - len(request_times)
            allowed = True
        
        return allowed, remaining
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than the time window"""
        cutoff_time = current_time - self.window_seconds
        
        # Clean up old requests
        for identifier in list(self.requests.keys()):
            self.requests[identifier][:] = [
                t for t in self.requests[identifier] 
                if t > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[identifier]:
                del self.requests[identifier]
        
        self.last_cleanup = current_time
        logger.info(f"[RATE_LIMIT] Cleanup completed. Active users: {len(self.requests)}")
    
    def get_wait_time(self, identifier: str) -> int:
        """
        Get seconds until next request is allowed.
        
        Returns:
            Seconds to wait (0 if can request now)
        """
        current_time = time.time()
        request_times = self.requests[identifier]
        
        if len(request_times) < self.max_requests:
            return 0
        
        # Find oldest request in current window
        cutoff_time = current_time - self.window_seconds
        valid_requests = [t for t in request_times if t > cutoff_time]
        
        if len(valid_requests) < self.max_requests:
            return 0
        
        # Calculate wait time until oldest request expires
        oldest_request = min(valid_requests)
        wait_seconds = int(self.window_seconds - (current_time - oldest_request)) + 1
        
        return max(0, wait_seconds)
    
    def reset(self, identifier: str):
        """Reset rate limit for a specific identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
            logger.info(f"[RATE_LIMIT] Reset for {identifier}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limiter statistics"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        active_users = 0
        total_requests = 0
        
        for identifier, request_times in self.requests.items():
            recent_requests = [t for t in request_times if t > cutoff_time]
            if recent_requests:
                active_users += 1
                total_requests += len(recent_requests)
        
        return {
            "active_users": active_users,
            "total_requests_last_minute": total_requests,
            "max_requests_per_minute": self.max_requests,
            "tracked_users": len(self.requests)
        }


# Global rate limiter instance
rate_limiter = SimpleRateLimiter(max_requests=10, window_seconds=60)