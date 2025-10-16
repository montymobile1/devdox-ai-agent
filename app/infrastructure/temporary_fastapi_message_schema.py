from pydantic import model_validator
from fastapi_mail import MessageSchema as _MessageSchema, MultipartSubtypeEnum

# There is an issue with Fast Mail current `MessageSchema` source code that was reported and waiting to be pushed here:
# https://github.com/sabuhish/fastapi-mail/pull/237, this is a temporary fix till the solution is pushed


class MessageSchema(_MessageSchema):
	@model_validator(mode="after")
	@classmethod
	def validate_alternative_body(cls, values):
		"""
		Validate alternative_body field
		"""
		if (
				values.multipart_subtype != MultipartSubtypeEnum.alternative
				and values.alternative_body
		):
			values.alternative_body = None
		return values