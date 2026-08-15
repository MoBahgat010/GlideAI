import logging
import redis.asyncio as aioredis
from config import REDIS_URL

logger = logging.getLogger("server.db.redis")

try:
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connection established successfully.")
except ConnectionError as e:
    raise ConnectionError("Redis connection failed") from e
except Exception as e:
    raise Exception(f"Error initializing Redis client: {e}")

async def close_redis():
    await redis_client.close()
    logger.info("Redis connection closed.")
