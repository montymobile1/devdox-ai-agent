from enum import StrEnum


class RepoStatus(StrEnum):
	COMPLETED = "completed"
	PENDING = "pending"
	FAILED = "failed"