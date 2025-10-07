from typing import Optional

from pydantic import ConfigDict, Field

from app.infrastructure.mailing_service.models.base_models import EmailEnvelope, NonBlankStr, OutgoingHtmlEmail, \
	OutgoingTemplatedContextEmail, OutgoingTemplatedHTMLEmail, OutgoingTemplatedTextEmail, OutgoingTextEmail


class PreviewEmailEnvelope(EmailEnvelope):
    model_config = ConfigDict(frozen=True)

class PreviewOutgoingHtmlEmail(OutgoingHtmlEmail):
    model_config = ConfigDict(frozen=True)

    html_body_preview: Optional[NonBlankStr] = Field(None, description="Rendered HTML for preview")
    text_fallback_preview: Optional[NonBlankStr] = Field(None, description="Rendered text fallback for preview")


class PreviewOutgoingTextEmail(OutgoingTextEmail):
    model_config = ConfigDict(frozen=True)

    text_body_preview: Optional[NonBlankStr] = Field(None, description="Rendered text body for preview")

class PreviewOutgoingTemplatedContextEmail(OutgoingTemplatedContextEmail):
    model_config = ConfigDict(frozen=True)

class PreviewOutgoingTemplatedHTMLEmail(OutgoingTemplatedHTMLEmail):
    model_config = ConfigDict(frozen=True)

    html_template_preview: Optional[NonBlankStr] = Field(None, description="Rendered HTML from template/context")
    plain_template_fallback_preview: Optional[NonBlankStr] = Field(None, description="Rendered plain-text fallback")


class PreviewOutgoingTemplatedTextEmail(OutgoingTemplatedTextEmail):
    model_config = ConfigDict(frozen=True)

    plain_template_preview: Optional[NonBlankStr] = Field(None, description="Rendered plain text from template/context")
