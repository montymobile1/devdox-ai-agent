from app.config import settings
from app.infrastructure.mailing_service import EmailDispatcher, EmailDispatchOptions, FastAPIMailClient


def get_email_dispatcher():
	"""Dependency injection container"""
	
	fast_mail_client = FastAPIMailClient(
		settings=settings.mail,
		dry_run=False
	)
	
	email_options = EmailDispatchOptions(
		subject_prefix=None,
		redirect_all_to=[],
		always_bcc=[],
	)
	
	email_dispatcher = EmailDispatcher(
		client=fast_mail_client,
		options=email_options,
	)
	
	return email_dispatcher