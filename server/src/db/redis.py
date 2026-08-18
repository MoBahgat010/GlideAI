from __future__ import annotations

import json
import logging
from typing import Any, Optional
import redis
import redis.asyncio as aioredis

from config import REDIS_URL

logger = logging.getLogger("server.db.redis")


class RedisManager:
    _async_client: Optional[aioredis.Redis] = None
    _sync_client: Optional[redis.Redis] = None

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        if cls._async_client is None:
            cls._async_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            logger.info("Async Redis client connected to %s", REDIS_URL)
        return cls._async_client

    @classmethod
    def get_sync_client(cls) -> redis.Redis:
        if cls._sync_client is None:
            cls._sync_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("Sync Redis client connected to %s", REDIS_URL)
        return cls._sync_client

    @classmethod
    async def ping(cls) -> bool:
        try:
            client = cls.get_client()
            return await client.ping()
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    @classmethod
    async def close(cls) -> None:
        if cls._async_client is not None:
            try:
                await cls._async_client.aclose()
                logger.info("Async Redis connection closed.")
            except Exception as exc:
                logger.warning("Error closing async Redis connection: %s", exc)
            finally:
                cls._async_client = None

        if cls._sync_client is not None:
            try:
                cls._sync_client.close()
                logger.info("Sync Redis connection closed.")
            except Exception as exc:
                logger.warning("Error closing sync Redis connection: %s", exc)
            finally:
                cls._sync_client = None


redis_client = RedisManager.get_client()


async def close_redis() -> None:
    await RedisManager.close()


async def get_redis() -> aioredis.Redis:
    return RedisManager.get_client()

class RedisAuthUtils:
    @staticmethod
    async def store_refresh_token(user_id: str, token_id: str, ttl_seconds: int) -> bool:
        client = RedisManager.get_client()
        try:
            redis_key = f"refresh_token:{token_id}"
            await client.setex(redis_key, ttl_seconds, user_id)
            await client.sadd(f"user_refresh_tokens:{user_id}", token_id)
            return True
        except Exception as exc:
            logger.warning("Failed to store refresh token %s in Redis: %s", token_id, exc)
            return False

    @staticmethod
    async def get_refresh_token_user(token_id: str) -> Optional[str]:
        client = RedisManager.get_client()
        try:
            return await client.get(f"refresh_token:{token_id}")
        except Exception as exc:
            logger.warning("Failed to fetch refresh token %s from Redis: %s", token_id, exc)
            return None

    @staticmethod
    async def remove_refresh_token(user_id: str, token_id: str) -> bool:
        client = RedisManager.get_client()
        try:
            await client.delete(f"refresh_token:{token_id}")
            await client.srem(f"user_refresh_tokens:{user_id}", token_id)
            return True
        except Exception as exc:
            logger.warning("Failed to remove refresh token %s: %s", token_id, exc)
            return False

    @staticmethod
    async def revoke_all_user_refresh_tokens(user_id: str) -> int:
        client = RedisManager.get_client()
        try:
            tokens_key = f"user_refresh_tokens:{user_id}"
            token_ids = await client.smembers(tokens_key)
            if not token_ids:
                return 0
            pipe = client.pipeline()
            for tid in token_ids:
                pipe.delete(f"refresh_token:{tid}")
            pipe.delete(tokens_key)
            await pipe.execute()
            return len(token_ids)
        except Exception as exc:
            logger.warning("Failed to revoke all refresh tokens for user %s: %s", user_id, exc)
            return 0

    @staticmethod
    async def blacklist_access_token(jti: str, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return True
        client = RedisManager.get_client()
        try:
            await client.setex(f"blacklist:{jti}", ttl_seconds, "revoked")
            return True
        except Exception as exc:
            logger.warning("Failed to blacklist token JTI %s: %s", jti, exc)
            return False

    @staticmethod
    async def is_token_blacklisted(jti: str) -> bool:
        client = RedisManager.get_client()
        try:
            status = await client.get(f"blacklist:{jti}")
            return bool(status)
        except Exception as exc:
            logger.warning("Failed to check blacklist for JTI %s: %s", jti, exc)
            return False


class RedisCacheUtils:
    @staticmethod
    async def get_json(key: str) -> Optional[Any]:
        """Fetch and deserialize a JSON value."""
        client = RedisManager.get_client()
        try:
            data = await client.get(key)
            return json.loads(data) if data else None
        except Exception as exc:
            logger.warning("Error fetching JSON key %s: %s", key, exc)
            return None

    @staticmethod
    async def set_json(key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Serialize and store a JSON value with optional TTL."""
        client = RedisManager.get_client()
        try:
            payload = json.dumps(value)
            if ttl_seconds:
                await client.setex(key, ttl_seconds, payload)
            else:
                await client.set(key, payload)
            return True
        except Exception as exc:
            logger.warning("Error setting JSON key %s: %s", key, exc)
            return False

    @staticmethod
    async def delete_keys(*keys: str) -> int:
        """Delete one or more keys."""
        if not keys:
            return 0
        client = RedisManager.get_client()
        try:
            return await client.delete(*keys)
        except Exception as exc:
            logger.warning("Error deleting keys %s: %s", keys, exc)
            return 0
