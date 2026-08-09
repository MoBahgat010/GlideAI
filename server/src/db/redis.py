import logging
import redis.asyncio as aioredis
from config import REDIS_URL


logger = logging.getLogger("server.db.redis")

_redis_client: aioredis.Redis | None = None


async def get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        logger.info("Initializing async Redis client for working memory: %s", REDIS_URL)
        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def close_redis():
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed.")
