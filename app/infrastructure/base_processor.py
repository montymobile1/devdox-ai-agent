"""
Base processor class for all task processors
"""

import logging
from typing import Any, Dict
from abc import ABC, abstractmethod
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Base class for all task processors"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_dir = Path(settings.BASE_DIR)
        self._ensure_base_directory()

    def _ensure_base_directory(self) -> None:
        """Ensure base directory exists and is writable"""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self.base_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

        except OSError as e:
            logger.error(f"Cannot access base directory {self.base_dir}: {e}")
            raise FileNotFoundError(f"Cannot access base directory {self.base_dir}: {e}")

    @abstractmethod
    async def process(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the job and return results

        Args:
            job_data: Job data from the queue

        Returns:
            Dict with processing results
        """
        pass

    def validate_job_data(self, job_data: Dict[str, Any], required_fields: list) -> bool:
        """Validate that job_data contains required fields"""
        for field in required_fields:
            if field not in job_data:
                raise ValueError(f"Missing required field: {field}")
        return True