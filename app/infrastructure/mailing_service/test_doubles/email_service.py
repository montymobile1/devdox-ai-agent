from typing import Any, List, Optional

from pydantic import EmailStr

from app.infrastructure.mailing_service.models.base_preview_models import (
    PreviewOutgoingTemplatedHTMLEmail,
)
from app.infrastructure.mailing_service.test_doubles.base import FakeBase
from app.infrastructure.mailing_service.service.template_resolver import Template
from app.infrastructure.mailing_service import IEmailDispatcher


class SpyEmailDispatcher(FakeBase, IEmailDispatcher):
    """Wraps a real IEmailDispatcher to spy on calls and optionally inject exceptions."""

    def __init__(self, inner: IEmailDispatcher) -> None:
        super().__init__()
        self.inner = inner
        self.returned_previews: list[PreviewOutgoingTemplatedHTMLEmail | None] = []

    async def send_templated_html(
            self,
            to: List[EmailStr],
            template: Template,
            context: Optional[dict[str, Any]] = None,
            subject: Optional[str] = None,
            cc: Optional[List[EmailStr]] = None,
            bcc: Optional[List[EmailStr]] = None,
            reply_to: Optional[List[EmailStr]] = None,
            headers: Optional[dict[str, str]] = None,
    ) -> PreviewOutgoingTemplatedHTMLEmail | None:
        self._before(
            self.send_templated_html,
            subject=subject,
            to=list(to),
            template=template,
            context=context,
            cc=list(cc or []),
            bcc=list(bcc or []),
            reply_to=list(reply_to or []),
            headers=headers,
        )
        
        out = await self.inner.send_templated_html(
            subject=subject,
            to=to,
            template=template,
            context=context,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            headers=headers,
        )
        self.returned_previews.append(out)
        return out
