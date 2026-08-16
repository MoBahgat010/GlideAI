import logging
from typing import Optional
import os
import cloudinary
import cloudinary.uploader

from config import (
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_FOLDER,
)

logger = logging.getLogger("storage.cloudinary")

class CloudinaryStorage:
    def __init__(
        self,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        folder: Optional[str] = None,
    ):
        self.cloud_name = cloud_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.folder = folder

        try:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True,
            )
            logger.info("Cloudinary storage successfully initialized for folder: %s", self.folder)
        except Exception as e:
            raise ConnectionError(f"Cloudinary configuration failed: {e}")

    def upload_file(self, file_path: str, public_id: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for upload: {file_path}")

        filename = os.path.basename(file_path)
        response = cloudinary.uploader.upload(
            file_path,
            folder=self.folder,
            public_id=public_id,
            resource_type="auto",
            overwrite=True,
            use_filename=True,
            unique_filename=False,
        )
        secure_url = response.get("secure_url")
        if not secure_url:
            raise RuntimeError(f"Cloudinary upload failed for {filename}")
        logger.warning("Uploaded source file '%s' to Cloudinary: %s", filename, secure_url)
        return secure_url


cloudinary_client = CloudinaryStorage(CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_FOLDER)