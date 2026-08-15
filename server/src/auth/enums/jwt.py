from enum import Enum


class TokenStatus(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
