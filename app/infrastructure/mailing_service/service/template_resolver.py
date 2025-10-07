from enum import Enum
from typing import Dict, Optional, Type

from pydantic import BaseModel, Field

from app.infrastructure.mailing_service.models.base_models import NonBlankStr
from app.infrastructure.mailing_service.models.context_shapes import ProjectAnalysisFailure
from app.infrastructure.mailing_service.models.context_shapes import ProjectAnalysisSuccess


class Template(Enum):
	PROJECT_ANALYSIS_FAILURE = "PROJECT_ANALYSIS_FAILURE"
	PROJECT_ANALYSIS_SUCCESS = "PROJECT_ANALYSIS_SUCCESS"

class TemplateShape(BaseModel):
	template_name: str
	subject:str
	
	context_shape: Type = Field(
		default=None, description="The context model class for this template"
	)
	
	html_template: Optional[NonBlankStr] = Field(
		default=None, description="Name of the HTML template file name (e.g., analysis_summary.html)"
	)
	
	plain_template: Optional[NonBlankStr] = Field(
		default=None,
		description=(
			"Name of the plain text template or fallback template file name (e.g., analysis_summary.txt)"
		)
	)


TEMPLATE_SOURCE:Dict[str, TemplateShape] = {
	Template.PROJECT_ANALYSIS_FAILURE.value: TemplateShape(
		subject="Repository Analysis Has Failed",
		template_name=Template.PROJECT_ANALYSIS_FAILURE.value,
		html_template="project_analysis_failure.html",
		plain_template="project_analysis_failure.txt",
		context_shape=ProjectAnalysisFailure
	),
Template.PROJECT_ANALYSIS_SUCCESS.value: TemplateShape(
		subject="Repository Analysis Successful",
		template_name=Template.PROJECT_ANALYSIS_SUCCESS.value,
		html_template="project_analysis_success.html",
		plain_template="project_analysis_success.txt",
		context_shape=ProjectAnalysisSuccess
	)
}

class TemplateResolver:
	
	def __init__(self, template_repository=None):
		self._template_repository = template_repository if template_repository is not None else TEMPLATE_SOURCE
	
	def get_template_meta_by_name(self, name: Template) -> TemplateShape:
		shape = self._template_repository[name.value]
		return shape