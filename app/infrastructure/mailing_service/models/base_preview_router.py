from typing import Any, Optional, overload, Union

from app.infrastructure.mailing_service.models.base_models import EmailEnvelope, NonBlankStr, OutgoingHtmlEmail, \
	OutgoingTemplatedContextEmail, OutgoingTemplatedHTMLEmail, OutgoingTemplatedTextEmail, OutgoingTextEmail
from app.infrastructure.mailing_service.models.base_preview_models import PreviewEmailEnvelope, PreviewOutgoingHtmlEmail, \
	PreviewOutgoingTemplatedContextEmail, PreviewOutgoingTemplatedHTMLEmail, PreviewOutgoingTemplatedTextEmail, \
	PreviewOutgoingTextEmail

BaseEmailUnion = Union[
	EmailEnvelope,
	OutgoingHtmlEmail,
	OutgoingTextEmail,
	OutgoingTemplatedContextEmail,
	OutgoingTemplatedHTMLEmail,
	OutgoingTemplatedTextEmail,
]

PreviewEmailUnion = Union[
	PreviewEmailEnvelope,
	PreviewOutgoingHtmlEmail,
	PreviewOutgoingTextEmail,
	PreviewOutgoingTemplatedContextEmail,
	PreviewOutgoingTemplatedHTMLEmail,
	PreviewOutgoingTemplatedTextEmail,
]

@overload
def make_preview(email: OutgoingHtmlEmail, *,
                 html_body_preview: Optional[NonBlankStr] = ...,
                 text_fallback_preview: Optional[NonBlankStr] = ...) -> PreviewOutgoingHtmlEmail: ...

@overload
def make_preview(email: OutgoingTextEmail, *,
                 text_body_preview: Optional[NonBlankStr] = ...) -> PreviewOutgoingTextEmail: ...

@overload
def make_preview(email: OutgoingTemplatedHTMLEmail, *,
                 html_template_preview: Optional[NonBlankStr] = ...,
                 plain_template_fallback_preview: Optional[NonBlankStr] = ...) -> PreviewOutgoingTemplatedHTMLEmail: ...

@overload
def make_preview(email: OutgoingTemplatedTextEmail, *,
                 plain_template_preview: Optional[NonBlankStr] = ...) -> PreviewOutgoingTemplatedTextEmail: ...

@overload
def make_preview(email: OutgoingTemplatedContextEmail) -> PreviewOutgoingTemplatedContextEmail: ...

@overload
def make_preview(email: EmailEnvelope) -> PreviewEmailEnvelope: ...

def make_preview(email: BaseEmailUnion, **kwargs: Any) -> PreviewEmailUnion:
	"""
	Create an immutable Preview* instance from a concrete email model.
	Public API is fully typed via overloads above (no dict-of-random-keys).
	"""
	data = email.model_dump()  # internal copy
	
	if isinstance(email, OutgoingTemplatedHTMLEmail):
		return PreviewOutgoingTemplatedHTMLEmail(**data,
		                                         html_template_preview=kwargs.get("html_template_preview"),
		                                         plain_template_fallback_preview=kwargs.get("plain_template_fallback_preview"),
		                                         )
	
	if isinstance(email, OutgoingTemplatedTextEmail):
		return PreviewOutgoingTemplatedTextEmail(**data,
		                                         plain_template_preview=kwargs.get("plain_template_preview"),
		                                         )
	
	if isinstance(email, OutgoingTemplatedContextEmail):
		return PreviewOutgoingTemplatedContextEmail(**data)
	
	if isinstance(email, OutgoingHtmlEmail):
		return PreviewOutgoingHtmlEmail(**data,
		                                html_body_preview=kwargs.get("html_body_preview"),
		                                text_fallback_preview=kwargs.get("text_fallback_preview"),
		                                )
	
	if isinstance(email, OutgoingTextEmail):
		return PreviewOutgoingTextEmail(**data,
		                                text_body_preview=kwargs.get("text_body_preview"),
		                                )
	
	return PreviewEmailEnvelope(**data)