import logging
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import MONGODB_URL, MONGODB_DB_NAME

logger = logging.getLogger("server.db.mongo")

# Safe resolution of connection parameters
_mongo_url = MONGODB_URL or "mongodb://localhost:27017"
_mongo_db_name = MONGODB_DB_NAME or "enterprise-rag"

# Singleton PyMongo client & DB instance
mongo_client = MongoClient(_mongo_url)
mongo_db = mongo_client[_mongo_db_name]

# Singleton Motor async client & DB instance
async_mongo_client = AsyncIOMotorClient(_mongo_url)
async_mongo_db = async_mongo_client[_mongo_db_name]


class MongoManager:
    client: AsyncIOMotorClient = async_mongo_client
    db: AsyncIOMotorDatabase = async_mongo_db
    sync_client: MongoClient = mongo_client
    sync_db = mongo_db

    @classmethod
    async def connect(cls):
        logger.info("Connecting to MongoDB at %s", MONGODB_URL)
        try:
            await cls.db.users.create_index("username", unique=True)
            await cls.db.users.create_index("email", unique=True)
            await cls.db.users.create_index("google_id", unique=True, sparse=True)
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
        if cls.sync_client:
            cls.sync_client.close()
        logger.info("MongoDB connection closed.")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        return cls.db

    @classmethod
    def get_sync_db(cls):
        return cls.sync_db


async def get_database() -> AsyncIOMotorDatabase:
    """FastAPI Dependency for database injection."""
    return async_mongo_db
