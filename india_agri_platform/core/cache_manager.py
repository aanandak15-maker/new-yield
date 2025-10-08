"""
Advanced Cache Management System for India Agricultural Intelligence Platform
Multi-level caching with Redis and in-memory fallbacks for optimal performance
"""

import asyncio
import pickle
import json
import time
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import redis.asyncio as redis
from dataclasses import dataclass, asdict
from enum import Enum

from india_agri_platform.core.error_handling import error_handler, ExternalAPIError, CacheError

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

class CacheLevel(Enum):
    """Cache levels (L1 = Memory, L2 = Redis, L3 = Disk)"""
    L1_MEMORY = "memory"
    L2_REDIS = "redis"
    L3_DISK = "disk"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    tags: List[str] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'size_bytes': self.size_bytes,
            'tags': self.tags or []
        }

class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_order: List[str] = []  # For LRU tracking
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    await self.delete(key)
                    return None

                # Update access tracking
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()

                # Move to end for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                return entry.value
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                  tags: List[str] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()

            # Calculate approximate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value).encode('utf-8'))

            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.utcnow(),
                ttl_seconds=ttl_seconds or self.default_ttl,
                size_bytes=size_bytes,
                tags=tags or []
            )

            self.cache[key] = entry

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries"""
        async with self._lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                await self.delete(key)

            return len(expired_keys)

    async def _evict_lru(self):
        """Evict least recently used item"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            return {
                'entries': len(self.cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'average_size_bytes': total_size / len(self.cache) if self.cache else 0,
                'hit_rate_estimate': 0.0,  # Would need more tracking
                'access_order_length': len(self.access_order)
            }

class RedisCache:
    """Redis-based distributed cache"""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 default_ttl: int = 3600):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.redis_client = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # Handle binary data
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except Exception as e:
            error_handler.handle_error(ExternalAPIError(
                message=f"Failed to connect to Redis: {str(e)}",
                api_name="Redis",
                status_code=None
            ))
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._connected or not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(key)
            if data:
                # Deserialize
                return pickle.loads(data)
            return None

        except Exception as e:
            error_handler.handle_error(CacheError(
                message=f"Redis get failed for key {key}: {str(e)}",
                details={'operation': 'get', 'key': key}
            ))
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                  tags: List[str] = None) -> bool:
        """Set value in Redis cache"""
        if not self._connected or not self.redis_client:
            return False

        try:
            # Serialize
            data = pickle.dumps(value)

            # Set with TTL
            ttl = ttl_seconds or self.default_ttl
            success = await self.redis_client.setex(key, ttl, data)

            # Store tags if provided (for batch operations)
            if tags:
                for tag in tags:
                    await self._add_to_tag_set(tag, key)

            return bool(success)

        except Exception as e:
            error_handler.handle_error(CacheError(
                message=f"Redis set failed for key {key}: {str(e)}",
                details={'operation': 'set', 'key': key, 'ttl': ttl_seconds}
            ))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self._connected or not self.redis_client:
            return False

        try:
            # Remove from tag sets first
            await self._remove_from_all_tags(key)

            # Delete the key
            result = await self.redis_client.delete(key)
            return result > 0

        except Exception as e:
            error_handler.handle_error(CacheError(
                message=f"Redis delete failed for key {key}: {str(e)}",
                details={'operation': 'delete', 'key': key}
            ))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self._connected or not self.redis_client:
            return False

        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            return False

    async def clear_by_tag(self, tag: str) -> int:
        """Clear all keys with a specific tag"""
        if not self._connected or not self.redis_client:
            return 0

        try:
            tag_key = f"tag:{tag}"
            keys = await self.redis_client.smembers(tag_key)

            if not keys:
                return 0

            # Delete all tagged keys
            await self.redis_client.delete(*keys)
            # Delete the tag set
            await self.redis_client.delete(tag_key)

            return len(keys)

        except Exception as e:
            error_handler.handle_error(CacheError(
                message=f"Redis tag clear failed for tag {tag}: {str(e)}",
                details={'operation': 'clear_by_tag', 'tag': tag}
            ))
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self._connected or not self.redis_client:
            return {'connected': False}

        try:
            info = await self.redis_client.info()
            db_info = info.get('db0', {})

            return {
                'connected': True,
                'keys': db_info.get('keys', 0),
                'memory_used': info.get('used_memory', 0),
                'memory_peak': info.get('used_memory_peak', 0),
                'connections': info.get('connected_clients', 0),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }

        except Exception as e:
            return {'connected': False, 'error': str(e)}

    async def _add_to_tag_set(self, tag: str, key: str):
        """Add key to tag set for group operations"""
        if not self.redis_client:
            return

        try:
            tag_key = f"tag:{tag}"
            await self.redis_client.sadd(tag_key, key)
        except Exception:
            pass  # Don't fail the main operation

    async def _remove_from_all_tags(self, key: str):
        """Remove key from all tag sets"""
        if not self.redis_client:
            return

        try:
            # This would require scanning all tag keys - simplified for now
            # In production, maintain reverse index
            pass
        except Exception:
            pass

class MultiLevelCache:
    """Multi-level cache system (L1 Memory + L2 Redis)"""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.l1_cache = MemoryCache(max_size=1000, default_ttl=1800)  # 30 min default
        self.l2_cache = RedisCache(host=redis_host, port=redis_port, default_ttl=3600*24)  # 24 hours

        # Cache performance tracking
        self.hits = {'l1': 0, 'l2': 0, 'miss': 0}
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize the multi-level cache"""
        redis_connected = await self.l2_cache.connect()
        if not redis_connected:
            logger.warning("Redis not available - operating with L1 cache only")

        logger.info("Multi-level cache initialized")
        return True

    async def get(self, key: str) -> Optional[Any]:
        """Get value with multi-level cache lookup"""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            async with self._lock:
                self.hits['l1'] += 1
            return value

        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Populate L1 cache
            await self.l1_cache.set(key, value, ttl_seconds=1800)  # 30 min in L1
            async with self._lock:
                self.hits['l2'] += 1
            return value

        # Cache miss
        async with self._lock:
            self.hits['miss'] += 1
        return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                  strategy: CacheStrategy = CacheStrategy.TTL,
                  tags: List[str] = None) -> bool:
        """Set value in multi-level cache"""
        success = True

        # Always set in L1
        l1_success = await self.l1_cache.set(key, value, ttl_seconds=min(ttl_seconds or 1800, 1800), tags=tags)
        success = success and l1_success

        # Set in L2 if Redis is available
        if self.l2_cache._connected:
            l2_ttl = ttl_seconds or self.l2_cache.default_ttl
            l2_success = await self.l2_cache.set(key, value, ttl_seconds=l2_ttl, tags=tags)
            success = success and l2_success

        return success

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key)

        return l1_deleted or l2_deleted

    async def clear_tag(self, tag: str) -> int:
        """Clear all entries with a specific tag"""
        l1_cleared = 0  # Would need tag tracking in memory cache
        l2_cleared = await self.l2_cache.clear_by_tag(tag)

        return l1_cleared + l2_cleared

    async def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries"""
        l1_expired = await self.l1_cache.cleanup_expired()
        # L2 cache handles TTL automatically in Redis

        return {
            'l1_expired': l1_expired,
            'l2_expired': 0  # Redis handles automatically
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()

        total_requests = sum(self.hits.values())
        l1_hit_rate = self.hits['l1'] / total_requests if total_requests > 0 else 0
        l2_hit_rate = self.hits['l2'] / total_requests if total_requests > 0 else 0

        return {
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'performance': {
                'total_requests': total_requests,
                'l1_hits': self.hits['l1'],
                'l2_hits': self.hits['l2'],
                'misses': self.hits['miss'],
                'l1_hit_rate': round(l1_hit_rate, 3),
                'l2_hit_rate': round(l2_hit_rate, 3),
                'overall_hit_rate': round((self.hits['l1'] + self.hits['l2']) / total_requests, 3) if total_requests > 0 else 0
            }
        }

class CacheManager:
    """Central cache management with intelligent caching strategies"""

    def __init__(self):
        self.cache = MultiLevelCache()
        self.cache_strategies = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the cache system"""
        if self._initialized:
            return True

        success = await self.cache.initialize()
        if success:
            # Setup periodic cleanup
            asyncio.create_task(self._periodic_cleanup())
            self._initialized = True

        return success

    async def _periodic_cleanup(self):
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                stats = await self.cache.cleanup()

                if stats['l1_expired'] > 0:
                    logger.info(f"Cleaned up {stats['l1_expired']} expired L1 cache entries")

            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")

    @asynccontextmanager
    async def cached_operation(self, operation_name: str, ttl_seconds: int = 3600):
        """Context manager for cached operations"""
        cache_key = f"operation:{operation_name}:{hash(str(asyncio.current_task()))}"

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            yield cached_result
            return

        # Perform operation (to be filled by caller)
        result = None
        yield result

        # Cache the result if operation completed successfully
        if result is not None:
            await self.cache.set(cache_key, result, ttl_seconds=ttl_seconds,
                               tags=['operation', operation_name])

    def make_cache_key(self, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments"""
        # Include function name, args, and sorted kwargs
        key_parts = []

        # Get calling function name
        frame = asyncio.current_task().get_coro().cr_frame
        if frame and frame.f_code.co_name != 'make_cache_key':
            key_parts.append(frame.f_code.co_name)

        # Add positional args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))

        # Add sorted keyword args
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")

        # Create final key
        key_string = ":".join(key_parts)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def cache_api_response(self, url: str, params: Dict[str, Any],
                               response: Any, ttl_seconds: int = 3600) -> bool:
        """Cache API response with intelligent key generation"""
        # Generate key from URL and params
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"api:{hashlib.md5(f'{url}:{param_str}'.encode()).hexdigest()}"

        return await self.cache.set(
            cache_key,
            response,
            ttl_seconds=ttl_seconds,
            tags=['api', url.split('/')[-1], 'response']
        )

    async def get_cached_api_response(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response"""
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"api:{hashlib.md5(f'{url}:{param_str}'.encode()).hexdigest()}"

        return await self.cache.get(cache_key)

    async def cache_prediction_result(self, prediction_input: Dict[str, Any],
                                    prediction_result: Dict[str, Any],
                                    ttl_seconds: int = 7200) -> bool:
        """Cache prediction results for faster repeated requests"""
        # Generate key from prediction inputs
        input_key = json.dumps(prediction_input, sort_keys=True)
        cache_key = f"prediction:{hashlib.md5(input_key.encode()).hexdigest()}"

        return await self.cache.set(
            cache_key,
            prediction_result,
            ttl_seconds=ttl_seconds,
            tags=['prediction', 'result']
        )

    async def invalidate_prediction_cache(self, field_id: str) -> int:
        """Invalidate prediction cache for a specific field"""
        # This would need to be implemented with proper tagging
        # For now, return 0 as placeholder
        return 0

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions for global use
async def initialize_cache() -> bool:
    """Initialize the global cache system"""
    return await cache_manager.initialize()

async def get_cached_value(key: str) -> Optional[Any]:
    """Get value from cache"""
    return await cache_manager.cache.get(key)

async def set_cached_value(key: str, value: Any, ttl_seconds: int = 3600,
                          tags: List[str] = None) -> bool:
    """Set value in cache"""
    return await cache_manager.cache.set(key, value, ttl_seconds=ttl_seconds, tags=tags)

async def get_cache_stats() -> Dict[str, Any]:
    """Get cache performance statistics"""
    return await cache_manager.cache.get_stats()
