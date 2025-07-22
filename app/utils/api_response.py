from typing import Any, Dict, List, Optional, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.responses import JSONResponse

Serializable = Union[
    BaseModel,
    List[BaseModel],
    Any,
]


def serialize_api_response_data(data: Optional[Serializable]) -> Any:
    """
    Serializes API response data into primitive types for JSONResponse compatibility.

    This utility currently supports:
    - Single Pydantic BaseModel instance → converted via `.model_dump()`
    - List of Pydantic BaseModel instances → each converted via `.model_dump()`
    - Later: could support dict[str, BaseModel] or nested serialization if needed.

    Any other type is returned as-is and assumed to be JSON-serializable.

    Args:
        data (Any): The data to serialize for API response. Can be a Pydantic model,
                    a list of models, or any other value.

    Returns:
        Any: A serialized version of the input suitable for use in JSONResponse.
    #
    """

    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    elif isinstance(data, list) and all(isinstance(item, BaseModel) for item in data):
        data = [item.model_dump(mode="json") for item in data]
    elif isinstance(data, dict):
        return {key: serialize_api_response_data(value) for key, value in data.items()}

    return data  # Return everything else as-is


class APIResponse:
    """Utility class for standardized API responses."""

    @staticmethod
    def success(message: str, data: Optional[Serializable] = None) -> JSONResponse:
        """Generate a success response."""
        response = {"success": True, "message": message, "status_code": 200}

        if data is not None:
            response["data"] = serialize_api_response_data(data)

        return JSONResponse(content=jsonable_encoder(response), status_code=200)

    @staticmethod
    def error(
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400,
        debug: Optional[Any] = None,
        error_type: Optional[str] = None,
    ) -> JSONResponse:
        """Generate an error response."""
        response = {
            "success": False,
            "message": message,
            "status_code": status_code,
            "error_type": error_type,
        }

        if debug is not None:
            response["debug"] = debug

        if details is not None:
            response["details"] = details

        return JSONResponse(content=jsonable_encoder(response), status_code=status_code)

    @staticmethod
    def validation_error(message: str, details: Optional[list] = None) -> JSONResponse:
        """Generate a validation error response."""
        response = {"success": False, "message": message, "status_code": 422}
        if details is not None:
            response["validation_errors"] = details
        return JSONResponse(content=jsonable_encoder(response), status_code=422)
