from .client.client import FastAPIMailClient
from .models.context_shapes import ProjectAnalysisFailure
from .service.email_service import (
    EmailDispatcher,
    EmailDispatchOptions,
    IEmailDispatcher,
)
from .service.interfaces import IEmailDispatcher

from .exception import mail_exceptions
from .exception import exception_constants

from .models.base_models import (
    EmailEnvelope,
    OutgoingHtmlEmail,
    OutgoingTemplatedContextEmail,
    OutgoingTemplatedHTMLEmail,
    OutgoingTemplatedTextEmail,
    OutgoingTextEmail,
)
from .models.base_preview_models import (
    PreviewEmailEnvelope,
    PreviewOutgoingHtmlEmail,
    PreviewOutgoingTemplatedContextEmail,
    PreviewOutgoingTemplatedHTMLEmail,
    PreviewOutgoingTemplatedTextEmail,
    PreviewOutgoingTextEmail,
)
from .service.template_resolver import Template, TemplateResolver

from .test_doubles.client import SpyMailClient
from .test_doubles.email_service import SpyEmailDispatcher


__all__ = [
    
    # client/
    "FastAPIMailClient",
    
    # service/
    "IEmailDispatcher",
    "EmailDispatcher",
    "EmailDispatchOptions",
    "Template",
    "TemplateResolver",
    "IEmailDispatcher",
    
    # exception/
    "mail_exceptions",
    "exception_constants",

    # models/
    "EmailEnvelope",
    "OutgoingHtmlEmail",
    "OutgoingTemplatedContextEmail",
    "OutgoingTemplatedHTMLEmail",
    "OutgoingTemplatedTextEmail",
    "OutgoingTextEmail",
    "PreviewEmailEnvelope",
    "PreviewOutgoingHtmlEmail",
    "PreviewOutgoingTemplatedContextEmail",
    "PreviewOutgoingTemplatedHTMLEmail",
    "PreviewOutgoingTemplatedTextEmail",
    "PreviewOutgoingTextEmail",
    "ProjectAnalysisFailure",
    
    # test_doubles/
    "SpyMailClient",
    "SpyEmailDispatcher"
]





