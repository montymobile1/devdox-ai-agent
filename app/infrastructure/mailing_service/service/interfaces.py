from abc import abstractmethod
from typing import  List, Optional, Protocol

from pydantic import EmailStr

from ..models.base_preview_models import PreviewOutgoingTemplatedHTMLEmail
from .template_resolver import Template
from ..models.context_shapes import BaseContextShape


class IEmailDispatcher(Protocol):
    @abstractmethod
    async def send_templated_html(
            self,
            to: List[EmailStr],
            template: Template,
            context: Optional[BaseContextShape] = None,
            subject: Optional[str] = None,
            cc: Optional[List[EmailStr]] = None,
            bcc: Optional[List[EmailStr]] = None,
            reply_to: Optional[List[EmailStr]] = None,
            headers: Optional[dict[str, str]] = None,
    ) -> PreviewOutgoingTemplatedHTMLEmail | None: ...
