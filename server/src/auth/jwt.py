from datetime import datetime, timedelta, timezone
import uuid
import logging
from typing import Optional, Tuple
import bcrypt
import jwt
from .enums.jwt import TokenStatus
from config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
)

from src.db.redis import redis_client

logger = logging.getLogger("server.auth.jwt")


def hash_password(password: str) -> str:
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pw_bytes = plain_password.encode("utf-8")
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pw_bytes, hash_bytes)
    except Exception:
        return False


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": TokenStatus.ACCESS.value,
        "jti": str(uuid.uuid4()),
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def create_refresh_token(user_id: str, username: str) -> Tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires_delta = timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    expire = now + expires_delta
    token_id = str(uuid.uuid4())

    payload = {
        "sub": username,
        "user_id": str(user_id),
        "jti": token_id,
        "type": TokenStatus.REFRESH.value,
        "iat": now,
        "exp": expire,
    }
    encoded_jwt = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    ttl_seconds = int(expires_delta.total_seconds())
    try:
        redis_key = f"refresh_token:{token_id}"
        await redis_client.setex(redis_key, ttl_seconds, user_id)
        await redis_client.sadd(f"user_refresh_tokens:{user_id}", token_id)
    except Exception as exc:
        logger.warning("Failed to store refresh token in Redis: %s", exc)

    return encoded_jwt, ttl_seconds


async def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_type = payload.get("type")
        if token_type not in (TokenStatus.ACCESS.value, TokenStatus.ACCESS, "access"):
            logger.warning("Token type mismatch: %r", token_type)
            return None

        jti = payload.get("jti")
        if jti:
            if await redis_client.get(f"blacklist:{jti}"):
                logger.warning("Access token jti %s is blacklisted", jti)
                return None

        return payload
    except Exception as exc:
        logger.warning("Failed to decode access token: %s", exc)
        return None


async def verify_and_rotate_refresh_token(refresh_token: str) -> Optional[Tuple[str, str, str]]:
    try:
        payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != TokenStatus.REFRESH.value:
            return None

        token_id = payload.get("jti")
        user_id = payload.get("user_id")
        username = payload.get("sub")

        if not token_id or not user_id or not username:
            return None

        redis_key = f"refresh_token:{token_id}"
        stored_user = await redis_client.get(redis_key)

        if not stored_user or stored_user != user_id:
            logger.warning("Refresh token not found or invalidated in Redis: %s", token_id)
            return None

        await redis_client.delete(redis_key)
        await redis_client.srem(f"user_refresh_tokens:{user_id}", token_id)

        new_refresh_token, _ = await create_refresh_token(user_id=user_id, username=username)
        return user_id, username, new_refresh_token
    except Exception as exc:
        logger.warning("Refresh token verification failed: %s", exc)
        return None


async def revoke_refresh_token(refresh_token: str) -> bool:
    try:
        payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_id = payload.get("jti")
        user_id = payload.get("user_id")
        if token_id and user_id:
            await redis_client.delete(f"refresh_token:{token_id}")
            await redis_client.srem(f"user_refresh_tokens:{user_id}", token_id)
            return True
    except Exception as exc:
        logger.warning("Revoke refresh token failed: %s", exc)
    return False


async def blacklist_access_token(access_token: str) -> None:
    try:
        payload = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti and exp:
            now_ts = datetime.now(timezone.utc).timestamp()
            remaining_ttl = int(exp - now_ts)
            if remaining_ttl > 0:
                await redis_client.setex(f"blacklist:{jti}", remaining_ttl, "revoked")
    except Exception as exc:
        logger.debug("Failed to blacklist access token: %s", exc)
