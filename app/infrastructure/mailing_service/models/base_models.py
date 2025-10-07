from typing import Annotated, Any, Mapping, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator, StringConstraints

from app.infrastructure.mailing_service.exception.exception_constants import CANNOT_SHARE_ADDRESS

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

def normalize_email(e: EmailStr) -> str:
	return str(e).strip().lower()

def dedupe(seq: list[EmailStr]) -> list[EmailStr]:
	"""Order-preserving, case-insensitive de-dupe."""
	seen: set[str] = set()
	out: list[EmailStr] = []
	for e in seq or []:
		k = normalize_email(e)
		if k in seen:
			continue
		seen.add(k)
		out.append(e)
	return out

class EmailEnvelope(BaseModel):
	subject: NonBlankStr = Field(default=..., description="The email’s title (what shows in the inbox). Keep it short and human (avoid ALL CAPS/spammy words).")
	recipients: list[EmailStr] = Field(default=..., min_length=1, description="A list of “To” addresses (the main people getting the email).")
	cc: list[EmailStr] = Field(default_factory=list, description="Carbon copy recipients")
	bcc: list[EmailStr] = Field(default_factory=list,  description="Blind carbon copy recipients")
	reply_to: list[EmailStr] = Field(default_factory=list, description="If someone clicks “Reply”, messages go here instead of the “From” address. Useful when sending from a no-reply address but wanting replies to support@…")
	headers: Optional[Mapping[str, NonBlankStr]] = Field(default=None, description="Custom metadata for filters/automation (machines more than humans). (e.g., 'X-Campaign-ID': 'llm-summary-2025-08')")
	
	# De-dupe each list independently
	@field_validator("recipients", "cc", "bcc", "reply_to", mode="after")
	@classmethod
	def _dedupe_each(cls, v: list[EmailStr]) -> list[EmailStr]:
		return dedupe(v)
	
	@model_validator(mode="after")
	def _no_cross_list_overlap(self) -> "EmailEnvelope":
		"""Forbid overlaps between To, Cc, and Bcc (case-insensitive)."""
		keyset = lambda xs: {normalize_email(x) for x in xs}
		to = keyset(self.recipients)
		cc = keyset(self.cc)
		bcc = keyset(self.bcc)
		
		if to & cc:
			raise ValueError(CANNOT_SHARE_ADDRESS.format(FROM = "To", TO="Cc"))
		if to & bcc:
			raise ValueError(CANNOT_SHARE_ADDRESS.format(FROM = "To", TO="Bcc"))
		if cc & bcc:
			raise ValueError(CANNOT_SHARE_ADDRESS.format(FROM = "Cc", TO="Bcc"))
		return self

class OutgoingHtmlEmail(EmailEnvelope):
	html_body: NonBlankStr = Field(default=..., description="Raw HTML body of the email")
	text_fallback: Optional[NonBlankStr] = Field(
		default=None, description="Optional text-only multipart fallback if HTML is not supported"
	)

class OutgoingTextEmail(EmailEnvelope):
	text_body: NonBlankStr = Field(default=..., description="Raw Plain text body of the email")

class OutgoingTemplatedContextEmail(EmailEnvelope):
	template_context: Optional[dict[str, Any]] = Field(
		default_factory=dict, description="Template context with variables for rendering (e.g., {'project_name': 'X', 'risks': [...]}))"
	)

class OutgoingTemplatedHTMLEmail(OutgoingTemplatedContextEmail):
	
	html_template: NonBlankStr = Field(
		default=..., description="Name of the HTML template file name (e.g., analysis_summary.html)"
	)
	
	plain_template_fallback: Optional[NonBlankStr] = Field(
		default=None, description="Optional text-only multipart fallback template file name if HTML is not supported"
	)

class OutgoingTemplatedTextEmail(OutgoingTemplatedContextEmail):
	plain_template: NonBlankStr = Field(
		default=..., description="Name of the plain text template file name (e.g., analysis_summary.txt)"
	)