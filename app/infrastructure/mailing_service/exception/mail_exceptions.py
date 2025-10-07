class MailError(RuntimeError):
    """Base class for all mail-layer errors."""

class MailConfigError(MailError):
    """Misconfiguration (e.g., templates disabled, bad path)."""

class MailTemplateError(MailError):
    """Template not found or render failure."""

class MailSendError(MailError):
    """SMTP/connect/send failure (including timeouts)."""