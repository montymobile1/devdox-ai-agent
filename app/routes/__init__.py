"""
Routes module initialization.
"""

from fastapi import APIRouter

from app.routes.analyze_code import router as analyze

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(analyze, tags=["AnalyzeRepo"])

