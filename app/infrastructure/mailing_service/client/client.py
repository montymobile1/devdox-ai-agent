import asyncio
from abc import abstractmethod
from typing import Any, Optional, Protocol

from fastapi_mail import ConnectionConfig, FastMail, MessageType, MultipartSubtypeEnum
from jinja2 import TemplateNotFound, TemplateSyntaxError

from app.config import MailSettings
from app.infrastructure.mailing_service.models.base_models import EmailEnvelope, OutgoingHtmlEmail, OutgoingTemplatedHTMLEmail, \
	OutgoingTemplatedTextEmail, OutgoingTextEmail
from app.infrastructure.mailing_service.models.base_preview_models import PreviewOutgoingHtmlEmail, \
	PreviewOutgoingTemplatedHTMLEmail, PreviewOutgoingTemplatedTextEmail, PreviewOutgoingTextEmail
from app.infrastructure.mailing_service.models.base_preview_router import make_preview
from app.infrastructure.mailing_service.exception import exception_constants
from app.infrastructure.mailing_service.exception.mail_exceptions import MailConfigError, MailSendError, MailTemplateError
from app.infrastructure.temporary_fastapi_message_schema import MessageSchema


class IMailClient(Protocol):
	@abstractmethod
	async def send_html_email(self, message: OutgoingHtmlEmail, timeout: int|None = None) -> PreviewOutgoingHtmlEmail | None: ...
	
	@abstractmethod
	async def send_text_email(self, message: OutgoingTextEmail, timeout: int|None = None) -> PreviewOutgoingTextEmail | None: ...
	
	@abstractmethod
	async def send_templated_html_email(self, message: OutgoingTemplatedHTMLEmail, timeout: int|None = None) ->  PreviewOutgoingTemplatedHTMLEmail | None: ...
	
	@abstractmethod
	async def send_templated_plain_email(self, message: OutgoingTemplatedTextEmail, timeout: int|None = None) -> PreviewOutgoingTemplatedTextEmail | None: ...

class FastAPIMailClient(IMailClient):
	
	def __init__(self, settings: MailSettings, dry_run: bool = False):
		"""
			:param settings: The settings to use.
			:param dry_run: If True, render templates and return a preview payload instead of sending.
		       Rendering follows the exact same code path as real sends and will raise on
		       template/config errors (so dev/test behaves like prod, without the SMTP server functionality).
		"""

		self.dry_run = dry_run
		
		self.is_templates_enabled = settings.templates_enabled
		
		self.send_timeout_seconds = settings.MAIL_SEND_TIMEOUT
		
		self.minimum_timeout_seconds = settings.MAIL_SEND_TIMEOUT_MIN
		
		self.conf = ConnectionConfig(
			MAIL_USERNAME=settings.MAIL_USERNAME,
			MAIL_PASSWORD=settings.MAIL_PASSWORD,
			MAIL_FROM=settings.MAIL_FROM,
			MAIL_FROM_NAME=settings.MAIL_FROM_NAME,
			MAIL_PORT=settings.MAIL_PORT,
			MAIL_SERVER=settings.MAIL_SERVER,
			MAIL_STARTTLS=settings.MAIL_STARTTLS,
			MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
			USE_CREDENTIALS=settings.MAIL_USE_CREDENTIALS,
			VALIDATE_CERTS=settings.MAIL_VALIDATE_CERTS,
			SUPPRESS_SEND=int(settings.MAIL_SUPPRESS_SEND),
			MAIL_DEBUG=settings.MAIL_DEBUG,
			TEMPLATE_FOLDER=settings.templates_dir
		)
		self._fm = FastMail(self.conf)
	
	@property
	def fm(self) -> FastMail:
		return self._fm
	
	# ---------- Internal operations ----------
	
	def _generate_message_schema(self, envelope:EmailEnvelope, subtype:MessageType) -> MessageSchema:
		headers = dict(envelope.headers) if envelope.headers else None
		
		return MessageSchema(
			subject=envelope.subject,
			recipients=envelope.recipients,
			cc=envelope.cc,
			bcc=envelope.bcc,
			reply_to=envelope.reply_to,
			headers=headers,
			subtype=subtype
		)
	
	def _ensure_templates_enabled(self) -> None:
		if not self.is_templates_enabled or not self.conf.TEMPLATE_FOLDER:
			raise MailConfigError(exception_constants.TEMPLATE_FOLDER_NOT_CONFIGURED)
	
	async def _send_fast_mail(self, message: MessageSchema, template_name: str = None, timeout: int|None = None) -> None:
		effective_timeout = None
		print("Sending email ..... ")
		try:
			# timeout guard so sends don’t hang forever
			effective_timeout = timeout if (timeout is not None and timeout > self.minimum_timeout_seconds) else self.send_timeout_seconds
			
			await asyncio.wait_for(
				self._fm.send_message(message=message, template_name=template_name),
				timeout=effective_timeout,
			)
			return None
		except asyncio.TimeoutError as e:
			raise MailSendError(
				f"SMTP send timed out after {effective_timeout}s "
				f"(subject='{message.subject}', server='{self.conf.MAIL_SERVER}:{self.conf.MAIL_PORT}')"
			) from e
		except Exception as e:
			raise MailSendError(
				f"SMTP send failed "
				f"(subject='{message.subject}', to={len(message.recipients)}, "
				f"cc={len(message.cc)}, bcc={len(message.bcc)}, "
				f"server='{self.conf.MAIL_SERVER}:{self.conf.MAIL_PORT}')"
			) from e
	
	# ---------- Template rendering helpers (used by both real + dry-run) ----------
	
	async def _render_template(
			self, template_name: str, context: dict[str, Any] | None
	) -> str:
		"""
		Render a Jinja template using FastMail's configured environment.
		Mirrors FastMail.__template_message_builder behavior for dict/list bodies.
		"""
		self._ensure_templates_enabled()
		try:
			env = self.conf.template_engine()
			tmpl = await self._fm.get_mail_template(env, template_name)  # Jinja2 Template
			
			data = context or {}
			if isinstance(data, list):
				# Keep parity with FastMail behavior for list bodies
				return tmpl.render({"body": data})
			if not isinstance(data, dict):
				raise ValueError(exception_constants.INVALID_TEMPLATE_BODY_TYPE)
			return tmpl.render(**data)
		except TemplateNotFound as e:
			raise MailTemplateError(exception_constants.TEMPLATE_NOT_FOUND.format(template_name=template_name)) from e
		except TemplateSyntaxError as e:
			raise MailTemplateError(exception_constants.TEMPLATE_SYNTAX_ERROR.format(name=e.name, lineno=e.lineno, message=e.message)) from e
		except Exception as e:
			raise MailTemplateError(exception_constants.TEMPLATE_RENDER_FAILED.format(template_name=template_name)) from e
	
	# ---------- Send Methods ----------
	
	async def send_html_email(self, message: OutgoingHtmlEmail, timeout: int|None = None) -> PreviewOutgoingHtmlEmail | None:
		
		if self.dry_run:
			return make_preview(
				message,
				html_body_preview=message.html_body,
				text_fallback_preview=message.text_fallback
			)
		
		message_schema = self._generate_message_schema(envelope=message, subtype=MessageType.html)
		message_schema.body = message.html_body
		
		if message.text_fallback:
			message_schema.alternative_body = message.text_fallback
			message_schema.multipart_subtype= MultipartSubtypeEnum.alternative
		
		return await self._send_fast_mail(message=message_schema, timeout=timeout)

	
	async def send_text_email(self, message: OutgoingTextEmail, timeout: int|None = None) -> PreviewOutgoingTextEmail | None:
		
		if self.dry_run:
			return make_preview(
				message,
				text_body_preview=message.text_body
			)
		
		message_schema = self._generate_message_schema(envelope=message, subtype=MessageType.plain)
		message_schema.body=message.text_body
		
		return await self._send_fast_mail(message=message_schema, timeout=timeout)
	
	async def send_templated_html_email(self, message: OutgoingTemplatedHTMLEmail, timeout: int|None = None) -> PreviewOutgoingTemplatedHTMLEmail | None:
		self._ensure_templates_enabled()
		
		text_fallback: Optional[str] = None
		
		if message.plain_template_fallback:
			text_fallback = await self._render_template(
				message.plain_template_fallback, message.template_context
			)
		
		if self.dry_run:
			html = await self._render_template(message.html_template, message.template_context or {})
			return make_preview(
				message,
				html_template_preview=html,
				plain_template_fallback_preview=text_fallback,
			)
		
		message_schema = self._generate_message_schema(envelope=message, subtype=MessageType.html)
		message_schema.template_body=message.template_context or {}
		
		if message.plain_template_fallback:
			message_schema.alternative_body=text_fallback
			message_schema.multipart_subtype=MultipartSubtypeEnum.alternative
		
		return await self._send_fast_mail(message=message_schema, template_name=message.html_template, timeout=timeout)
	
	async def send_templated_plain_email(self, message: OutgoingTemplatedTextEmail, timeout: int|None = None) -> PreviewOutgoingTemplatedTextEmail | None:
		self._ensure_templates_enabled()
		
		text_body = await self._render_template(
			message.plain_template, message.template_context
		)
		
		if self.dry_run:
			return make_preview(
				message,
				plain_template_preview=text_body
			)
		
		message_schema = self._generate_message_schema(envelope=message, subtype=MessageType.plain)
		
		message_schema.body=text_body
		
		return await self._send_fast_mail(message=message_schema, timeout=timeout)