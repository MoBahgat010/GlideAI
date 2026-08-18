import logging
from pymongo import MongoClient
from pymongo.database import Database
from config import MONGODB_URL, MONGODB_DB_NAME

logger = logging.getLogger("server.db.mongo")


class MongoDB:
    """Centralized singleton PyMongo client and database manager."""

    def __init__(self, url: str = MONGODB_URL, db_name: str = MONGODB_DB_NAME):
        self.url = url or "mongodb://localhost:27017"
        self.db_name = db_name or "enterprise-rag"
        self.client: MongoClient = MongoClient(self.url)
        self.db: Database = self.client[self.db_name]

    @property
    def users(self):
        return self.db.users

    @property
    def sessions(self):
        return self.db.sessions

    def __getitem__(self, collection_name: str):
        return self.db[collection_name]

    def __getattr__(self, item: str):
        return getattr(self.db, item)

    def connect(self):
        """Create and verify database indexes on startup."""
        logger.info("Initializing MongoDB at %s (%s)", self.url, self.db_name)
        self.users.create_index("username", unique=True)
        self.users.create_index("email", unique=True)
        self.users.create_index("google_id", unique=True, sparse=True)
        self.sessions.create_index("session_id", unique=True)
        self.sessions.create_index("user_id")
        logger.info("MongoDB connection and indexes verified successfully.")

    def close(self):
        """Close MongoDB client connection."""
        self.client.close()
        logger.info("MongoDB connection closed.")

    def get_db(self) -> Database:
        return self.db


mongo = MongoDB()
