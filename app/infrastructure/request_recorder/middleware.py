import base64
import json
import logging
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TypedDict, Any

from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.exceptions import HTTPException

from encryption_src.fernet.service import FernetEncryptionHelper
from models_src.dto.api_log import ApiLogRequestDTO
from models_src.repositories.api_log import TortoiseApiLogStore
from models_src.repositories.user import TortoiseUserStore
from app.utils.encryption import get_encryption_helper

logger = logging.getLogger(__name__)


class IncomingRequest(TypedDict):
    operation_id: str
    user_id: str
    request_received_at: datetime
    process_time_ms: int


class OperationLogRule(TypedDict):
    """
    Required per-operation rule.
    - operation_id: FastAPI route.operation_id to include in logging
    - redact_keys: keys to redact (case-insensitive) from request/response JSON
    """
    operation_id: str
    redact_keys: list[str] | set[str]


class RecordRequestMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        include_rules: list[OperationLogRule] | None,   # REQUIRED allowlist of ops to log
        encrypt_bodies: bool = True,                    # GLOBAL: if True, require user salt and encrypt
        api_log_store: TortoiseApiLogStore | None = None,
        user_store: TortoiseUserStore | None = None,
        encryptor: FernetEncryptionHelper | None = None,
    ):
        if not include_rules:
            raise ValueError("'include_rules' is required and cannot be empty.")

        super().__init__(app)
        self.api_log_store = api_log_store or TortoiseApiLogStore()
        self.user_store = user_store or TortoiseUserStore()
        self.encryptor = encryptor or get_encryption_helper()
        self.encrypt_bodies = encrypt_bodies

        # Map op_id -> normalized rule
        self._rules_by_op: dict[str, OperationLogRule] = {}
        for r in include_rules:
            op = r.get("operation_id")
            if not op or not isinstance(op, str):
                raise ValueError("Each include rule must have a valid 'operation_id' (str).")
            redact = r.get("redact_keys") or []
            self._rules_by_op[op] = {
                "operation_id": op,
                "redact_keys": {k.lower() for k in redact},
            }

    # ---------- helpers ----------

    def _parse_json(self, raw: bytes) -> Any | None:
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _redact(self, data: Any, redact_keys: set[str]) -> Any:
        if not redact_keys or data is None:
            return data
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(k, str) and k.lower() in redact_keys:
                    out[k] = "***"
                else:
                    out[k] = self._redact(v, redact_keys)
            return out
        if isinstance(data, list):
            return [self._redact(x, redact_keys) for x in data]
        return data

    async def _get_required_user_salt(self, user_sub: str | None) -> str:
        """Fail-fast: raise if user salt is missing or undecryptable."""
        if not user_sub:
            logger.error("Missing user_sub; encryption is enabled and user salt is required.")
            raise HTTPException(status_code=500, detail="Missing user identity for encryption.")
        try:
            enc_salt = await self.user_store.get_encryption_salt(user_sub)
        except Exception:
            logger.exception("Failed retrieving user salt for sub=%s", user_sub)
            raise HTTPException(status_code=500, detail="Failed retrieving user encryption salt.")
        if not enc_salt:
            logger.error("User salt not found for sub=%s", user_sub)
            raise HTTPException(status_code=500, detail="User encryption salt not found.")
        try:
            return self.encryptor.decrypt(enc_salt)
        except Exception:
            logger.exception("Failed to decrypt user salt for sub=%s", user_sub)
            raise HTTPException(status_code=500, detail="Failed to decrypt user encryption salt.")
    
    def _to_encryptable_text(self, obj: Any) -> str:
        """
        Normalize any JSON-serializable or raw payload to a string so
        FernetEncryptionHelper.encrypt_for_user(plaintext: str, salt_b64: str) can handle it.
        """
        if isinstance(obj, (dict, list)):
            # Stable, compact JSON
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if isinstance(obj, (int, float, bool)) or obj is None:
            # Keep JSON semantics for primitives
            return json.dumps(obj, ensure_ascii=False)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (bytes, bytearray, memoryview)):
            b = bytes(obj)
            # Prefer UTF-8 text if possible; otherwise base64 to preserve data
            try:
                return b.decode("utf-8")
            except UnicodeDecodeError:
                return "base64:" + base64.b64encode(b).decode("ascii")
        # Last resort: stringify with JSON to avoid repr() surprises
        return json.dumps(obj, default=str, ensure_ascii=False)
    
    def _encrypt_or_plain(self, obj: Any, *, user_salt: str | None) -> Any | None:
        if obj is None:
            return None
        if not self.encrypt_bodies:
            return obj

        plaintext = self._to_encryptable_text(obj)
        return { "value": self.encryptor.encrypt_for_user(plaintext, user_salt) }


    async def _write_log(self, payload: dict):
        try:
            await self.api_log_store.save(ApiLogRequestDTO(**payload))
        except Exception:
            logger.exception("Failed saving ApiLogRequestDTO (operation_id=%s)", payload.get("operation_id"))

    # ---------- middleware ----------

    async def dispatch(self, request: Request, call_next):
        start_ns = time.perf_counter_ns()
        ts_server_received = datetime.now(timezone.utc)
        
        # Read request body early (Starlette caches for downstream)
        try:
            raw_req_body = await request.body()
        except Exception:
            logger.exception("Failed reading request body")
            raw_req_body = b""

        response = await call_next(request)

        # Route metadata becomes available only after routing
        route = request.scope.get("route")
        operation_id = getattr(route, "operation_id", None)
        
        rule = self._rules_by_op.get(operation_id)
        # If op not included, just return response (only log included ops)
        if not rule:
            return response
        
        # Identify user and fail-fast on salt if encryption is enabled
        claims = getattr(request.state, "user_claims", None) or SimpleNamespace()
        user_sub = getattr(claims, "sub", "") or ""
        user_salt: str | None = None
        if self.encrypt_bodies:
            user_salt = await self._get_required_user_salt(user_sub)

        # Buffer response to both log and send intact
        try:
            chunks = [section async for section in response.body_iterator]
            body_bytes = b"".join(chunks)
            response.body_iterator = iterate_in_threadpool(iter([body_bytes]))
        except Exception:
            logger.exception("Failed buffering response body (operation_id=%s)", operation_id)
            return response

        # Timing
        incoming_request: IncomingRequest = {
            "operation_id": operation_id,
            "user_id": user_sub,
            "request_received_at": ts_server_received,
            "process_time_ms": (time.perf_counter_ns() - start_ns + 500_000) // 1_000_000,
        }

        # Parse + redact
        try:
            req_json = self._parse_json(raw_req_body)
            res_json = self._parse_json(body_bytes)
            redact_keys = rule["redact_keys"]
            req_json = self._redact(req_json, redact_keys)
            res_json = self._redact(res_json, redact_keys)
        except Exception:
            logger.exception("Parse/redaction failed (operation_id=%s)", operation_id)
            req_json = res_json = None

        # Encrypt or keep plaintext. user_salt is guaranteed if encrypt_bodies=True.
        try:
            enc_req = self._encrypt_or_plain(req_json, user_salt=user_salt)
            enc_res = self._encrypt_or_plain(res_json, user_salt=user_salt)
        except HTTPException:
            raise  # re-raise if you flipped to strict fail on _encrypt_or_plain
        except Exception:
            logger.exception("Body prepare failed (operation_id=%s)", operation_id)
            enc_req = enc_res = None

        dto_payload = {
            "operation_id": operation_id,
            "path": str(request.url.path),
            "method": request.method,
            "user_id": user_sub,
            "request_received_at": incoming_request["request_received_at"],
            "process_time_ms": incoming_request["process_time_ms"],
            "request_body": enc_req,
            "response_body": enc_res,
        }

        try:
            response.background = BackgroundTask(self._write_log, dto_payload)
        except Exception:
            logger.exception("Failed scheduling background log write (operation_id=%s)", operation_id)

        return response
