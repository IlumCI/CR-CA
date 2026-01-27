"""
Rate limiting utility module.

Provides rate limiting functionality for agents and API calls with support for:
- Token-based and request-based limiting
- Per-user/session rate limits
- Queue management for rate-limited requests
- Configurable limits (RPM, RPH)
"""

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
from loguru import logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_minute: Maximum requests per minute (0 = unlimited)
        requests_per_hour: Maximum requests per hour (0 = unlimited)
        tokens_per_minute: Maximum tokens per minute (0 = unlimited)
        tokens_per_hour: Maximum tokens per hour (0 = unlimited)
        queue_enabled: Whether to queue requests when rate limited
        max_queue_size: Maximum queue size
    """
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 0  # 0 = unlimited
    tokens_per_hour: int = 0  # 0 = unlimited
    queue_enabled: bool = True
    max_queue_size: int = 1000


class RateLimiter:
    """Rate limiter with per-user/session tracking.
    
    Provides rate limiting functionality with support for:
    - Request-based limiting (RPM, RPH)
    - Token-based limiting (TPM, TPH)
    - Per-user/session tracking
    - Queue management for rate-limited requests
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration. Uses defaults if None.
        """
        self.config = config or RateLimitConfig()
        self._lock = threading.RLock()
        
        # Per-user tracking: {user_id: {type: deque of timestamps}}
        self._request_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "minute": deque(maxlen=self.config.requests_per_minute * 2),
                "hour": deque(maxlen=self.config.requests_per_hour * 2),
            }
        )
        
        self._token_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "minute": deque(maxlen=1000),
                "hour": deque(maxlen=10000),
            }
        )
        
        # Queue for rate-limited requests
        self._queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_queue_size))
        
        logger.debug(f"Initialized RateLimiter with config: {self.config}")
    
    def _clean_old_entries(self, user_id: str, window: str) -> None:
        """Remove old entries outside the time window.
        
        Args:
            user_id: User identifier
            window: Time window ('minute' or 'hour')
        """
        current_time = time.time()
        cutoff = current_time - (60 if window == "minute" else 3600)
        
        # Clean request history
        if user_id in self._request_history:
            history = self._request_history[user_id][window]
            while history and history[0] < cutoff:
                history.popleft()
        
        # Clean token history
        if user_id in self._token_history:
            history = self._token_history[user_id][window]
            while history and len(history) > 0:
                if history[0][0] < cutoff:
                    history.popleft()
                else:
                    break
    
    def check_rate_limit(
        self,
        user_id: str = "default",
        token_count: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """Check if request is within rate limits.
        
        Args:
            user_id: User identifier for per-user tracking
            token_count: Number of tokens in the request (0 if not applicable)
            
        Returns:
            Tuple of (is_allowed, error_message)
            - is_allowed: True if request is allowed, False otherwise
            - error_message: Error message if not allowed, None otherwise
        """
        with self._lock:
            current_time = time.time()
            
            # Clean old entries
            self._clean_old_entries(user_id, "minute")
            self._clean_old_entries(user_id, "hour")
            
            # Check request-based limits
            if self.config.requests_per_minute > 0:
                minute_requests = len(self._request_history[user_id]["minute"])
                if minute_requests >= self.config.requests_per_minute:
                    return False, f"Rate limit exceeded: {minute_requests}/{self.config.requests_per_minute} requests per minute"
            
            if self.config.requests_per_hour > 0:
                hour_requests = len(self._request_history[user_id]["hour"])
                if hour_requests >= self.config.requests_per_hour:
                    return False, f"Rate limit exceeded: {hour_requests}/{self.config.requests_per_hour} requests per hour"
            
            # Check token-based limits
            if token_count > 0:
                if self.config.tokens_per_minute > 0:
                    minute_tokens = sum(
                        tokens for _, tokens in self._token_history[user_id]["minute"]
                    )
                    if minute_tokens + token_count > self.config.tokens_per_minute:
                        return False, f"Token limit exceeded: {minute_tokens + token_count}/{self.config.tokens_per_minute} tokens per minute"
                
                if self.config.tokens_per_hour > 0:
                    hour_tokens = sum(
                        tokens for _, tokens in self._token_history[user_id]["hour"]
                    )
                    if hour_tokens + token_count > self.config.tokens_per_hour:
                        return False, f"Token limit exceeded: {hour_tokens + token_count}/{self.config.tokens_per_hour} tokens per hour"
            
            # Record request
            self._request_history[user_id]["minute"].append(current_time)
            self._request_history[user_id]["hour"].append(current_time)
            
            # Record tokens if applicable
            if token_count > 0:
                self._token_history[user_id]["minute"].append((current_time, token_count))
                self._token_history[user_id]["hour"].append((current_time, token_count))
            
            return True, None
    
    def wait_if_rate_limited(
        self,
        user_id: str = "default",
        token_count: int = 0,
        max_wait: float = 60.0,
    ) -> bool:
        """Wait if rate limited, up to max_wait seconds.
        
        Args:
            user_id: User identifier
            token_count: Number of tokens in the request
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if request can proceed, False if still rate limited after waiting
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            is_allowed, error_msg = self.check_rate_limit(user_id, token_count)
            if is_allowed:
                return True
            
            # Calculate wait time
            if "minute" in error_msg:
                # Wait until oldest request in minute window expires
                if self._request_history[user_id]["minute"]:
                    oldest = self._request_history[user_id]["minute"][0]
                    wait_time = min(60 - (time.time() - oldest) + 1, max_wait)
                    if wait_time > 0:
                        time.sleep(min(wait_time, 1.0))  # Sleep in 1s increments
                else:
                    time.sleep(0.1)
            elif "hour" in error_msg:
                # Wait until oldest request in hour window expires
                if self._request_history[user_id]["hour"]:
                    oldest = self._request_history[user_id]["hour"][0]
                    wait_time = min(3600 - (time.time() - oldest) + 1, max_wait)
                    if wait_time > 0:
                        time.sleep(min(wait_time, 1.0))
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        
        return False
    
    def get_rate_limit_status(self, user_id: str = "default") -> Dict[str, Any]:
        """Get current rate limit status for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with rate limit status information
        """
        with self._lock:
            self._clean_old_entries(user_id, "minute")
            self._clean_old_entries(user_id, "hour")
            
            minute_requests = len(self._request_history[user_id]["minute"])
            hour_requests = len(self._request_history[user_id]["hour"])
            
            minute_tokens = sum(
                tokens for _, tokens in self._token_history[user_id]["minute"]
            )
            hour_tokens = sum(
                tokens for _, tokens in self._token_history[user_id]["hour"]
            )
            
            return {
                "user_id": user_id,
                "requests": {
                    "minute": {
                        "current": minute_requests,
                        "limit": self.config.requests_per_minute,
                        "remaining": max(0, self.config.requests_per_minute - minute_requests),
                    },
                    "hour": {
                        "current": hour_requests,
                        "limit": self.config.requests_per_hour,
                        "remaining": max(0, self.config.requests_per_hour - hour_requests),
                    },
                },
                "tokens": {
                    "minute": {
                        "current": minute_tokens,
                        "limit": self.config.tokens_per_minute,
                        "remaining": max(0, self.config.tokens_per_minute - minute_tokens) if self.config.tokens_per_minute > 0 else -1,
                    },
                    "hour": {
                        "current": hour_tokens,
                        "limit": self.config.tokens_per_hour,
                        "remaining": max(0, self.config.tokens_per_hour - hour_tokens) if self.config.tokens_per_hour > 0 else -1,
                    },
                },
            }
    
    def reset_user(self, user_id: str) -> None:
        """Reset rate limit tracking for a user.
        
        Args:
            user_id: User identifier
        """
        with self._lock:
            if user_id in self._request_history:
                del self._request_history[user_id]
            if user_id in self._token_history:
                del self._token_history[user_id]
            if user_id in self._queues:
                del self._queues[user_id]
            logger.debug(f"Reset rate limit tracking for user: {user_id}")
    
    def reset_all(self) -> None:
        """Reset rate limit tracking for all users."""
        with self._lock:
            self._request_history.clear()
            self._token_history.clear()
            self._queues.clear()
            logger.debug("Reset rate limit tracking for all users")
