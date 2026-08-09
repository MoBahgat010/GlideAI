import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import MONGODB_URL, MONGODB_DB_NAME


logger = logging.getLogger("server.db.mongo")


class MongoManager:
    """Async MongoDB Connection Manager."""

    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

    @classmethod
    async def connect(cls):
        if cls.client is None:
            logger.info("Connecting to MongoDB at %s", MONGODB_URL)
            cls.client = AsyncIOMotorClient(MONGODB_URL)
            cls.db = cls.client[MONGODB_DB_NAME]
            # Ensure unique indexes
            try:
                await cls.db.users.create_index("username", unique=True)
                await cls.db.users.create_index("email", unique=True)
                await cls.db.sessions.create_index("session_id", unique=True)
                await cls.db.sessions.create_index("user_id")
                await cls.db.episodic_memories.create_index("session_id")
                await cls.db.semantic_memories.create_index("session_id")
                logger.info("MongoDB connection established and indexes verified.")
            except Exception as e:
                logger.warning("MongoDB index creation warning: %s", e)

    @classmethod
    async def close(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("MongoDB connection closed.")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        if cls.db is None:
            # Fallback inline connection
            cls.client = AsyncIOMotorClient(MONGODB_URL)
            cls.db = cls.client[MONGODB_DB_NAME]
        return cls.db


async def get_database() -> AsyncIOMotorDatabase:
    """FastAPI Dependency for database injection."""
    return MongoManager.get_db()
