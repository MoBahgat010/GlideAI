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
        self.cloud_name = cloud_name or CLOUDINARY_CLOUD_NAME
        self.api_key = api_key or CLOUDINARY_API_KEY
        self.api_secret = api_secret or CLOUDINARY_API_SECRET
        self.folder = folder or CLOUDINARY_FOLDER
        self.is_configured = bool(
            self.cloud_name and self.api_key and self.api_secret
        )

        if self.is_configured and cloudinary:
            try:
                cloudinary.config(
                    cloud_name=self.cloud_name,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    secure=True,
                )
                logger.info("Cloudinary storage successfully initialized for folder: %s", self.folder)
            except Exception as e:
                logger.warning("Cloudinary configuration failed: %s", e)
                self.is_configured = False
        else:
            logger.info("Cloudinary not configured; assets will use inline data URIs.")

    def upload_image_base64(self, base64_str: str, public_id: Optional[str] = None) -> Optional[str]:
        """Upload a base64 encoded image or data-URI to Cloudinary."""
        if not self.is_configured or not cloudinary:
            return None

        try:
            payload = base64_str if base64_str.startswith("data:image") else f"data:image/png;base64,{base64_str}"
            response = cloudinary.uploader.upload(
                payload,
                folder=self.folder,
                public_id=public_id,
                resource_type="image",
                overwrite=True,
            )
            secure_url = response.get("secure_url")
            logger.debug("Uploaded image to Cloudinary: %s", secure_url)
            return secure_url
        except Exception as exc:
            logger.warning("Cloudinary image upload failed for id=%s: %s", public_id, exc)
            return None

    def upload_file(self, file_path: str, public_id: Optional[str] = None) -> Optional[str]:
        """Upload a local file (e.g. PDF page, diagram, media) to Cloudinary."""
        if not self.is_configured or not cloudinary or not os.path.exists(file_path):
            return None

        try:
            response = cloudinary.uploader.upload(
                file_path,
                folder=self.folder,
                public_id=public_id,
                resource_type="auto",
                overwrite=True,
            )
            return response.get("secure_url")
        except Exception as exc:
            logger.warning("Cloudinary file upload failed for %s: %s", file_path, exc)
            return None


cloudinary_client = CloudinaryStorage()