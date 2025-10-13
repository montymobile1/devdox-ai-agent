"""
Routes module initialization.
"""

from fastapi import APIRouter

from app.routes.analyze_code import router as analyze
from app.routes.load_test import router as load_test
from app.routes.health_check import router as health_check

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(analyze, tags=["AnalyzeRepo"])
router.include_router(load_test, tags=["LoadTestRepo"])

router.include_router(health_check, tags=["HealthCheck"])

