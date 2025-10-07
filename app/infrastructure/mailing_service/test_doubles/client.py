from app.infrastructure.mailing_service.client.client import IMailClient
from app.infrastructure.mailing_service.models.base_models import (
    OutgoingHtmlEmail,
    OutgoingTextEmail,
    OutgoingTemplatedHTMLEmail,
    OutgoingTemplatedTextEmail,
)
from app.infrastructure.mailing_service.models.base_preview_models import (
    PreviewOutgoingHtmlEmail,
    PreviewOutgoingTextEmail,
    PreviewOutgoingTemplatedHTMLEmail,
    PreviewOutgoingTemplatedTextEmail,
)
from app.infrastructure.mailing_service.test_doubles.base import FakeBase


class SpyMailClient(FakeBase, IMailClient):
    """Wraps a real IMailClient to spy on calls and optionally inject exceptions."""

    def __init__(self, inner: IMailClient) -> None:
        super().__init__()
        self.inner = inner

        # Handy buckets you can assert on in tests
        self.captured_html: list[OutgoingHtmlEmail] = []
        self.captured_text: list[OutgoingTextEmail] = []
        self.captured_tpl_html: list[OutgoingTemplatedHTMLEmail] = []
        self.captured_tpl_text: list[OutgoingTemplatedTextEmail] = []

        self.returned_html: list[PreviewOutgoingHtmlEmail | None] = []
        self.returned_text: list[PreviewOutgoingTextEmail | None] = []
        self.returned_tpl_html: list[PreviewOutgoingTemplatedHTMLEmail | None] = []
        self.returned_tpl_text: list[PreviewOutgoingTemplatedTextEmail | None] = []

    async def send_html_email(
        self, message: OutgoingHtmlEmail, timeout: int | None = None
    ) -> PreviewOutgoingHtmlEmail | None:
        self._before(self.send_html_email, message=message, timeout=timeout)
        self.captured_html.append(message)
        out = await self.inner.send_html_email(message, timeout=timeout)
        self.returned_html.append(out)
        return out

    async def send_text_email(
        self, message: OutgoingTextEmail, timeout: int | None = None
    ) -> PreviewOutgoingTextEmail | None:
        self._before(self.send_text_email, message=message, timeout=timeout)
        self.captured_text.append(message)
        out = await self.inner.send_text_email(message, timeout=timeout)
        self.returned_text.append(out)
        return out

    async def send_templated_html_email(
        self, message: OutgoingTemplatedHTMLEmail, timeout: int | None = None
    ) -> PreviewOutgoingTemplatedHTMLEmail | None:
        self._before(self.send_templated_html_email, message=message, timeout=timeout)
        self.captured_tpl_html.append(message)
        out = await self.inner.send_templated_html_email(message, timeout=timeout)
        self.returned_tpl_html.append(out)
        return out

    async def send_templated_plain_email(
        self, message: OutgoingTemplatedTextEmail, timeout: int | None = None
    ) -> PreviewOutgoingTemplatedTextEmail | None:
        self._before(self.send_templated_plain_email, message=message, timeout=timeout)
        self.captured_tpl_text.append(message)
        out = await self.inner.send_templated_plain_email(message, timeout=timeout)
        self.returned_tpl_text.append(out)
        return out
