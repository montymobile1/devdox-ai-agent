"""
FastAPI application entry point for agent MCP server
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends
from fastapi_mcp import FastApiMCP, AuthConfig
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings, TORTOISE_ORM
from app.exceptions.register import register_exception_handlers
from app.logging_config import setup_logging
from app.routes import router as api_router
from app.utils.auth import mcp_auth_interceptor

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    from tortoise import Tortoise

    await Tortoise.init(config=TORTOISE_ORM)
    yield

    # Shutdown
    await Tortoise.close_connections()


# Initialize FastAPI app
app = FastAPI(
    title="DevDox AI Agent API",
    description="This component enables developers to automate code creation, maintenance, and documentation",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Register all exception handlers from one place
register_exception_handlers(app)


mcp = FastApiMCP(app,
                 name="My API MCP",
                 describe_all_responses=True,
                 describe_full_response_schema=True,
                 auth_config=AuthConfig(
                     dependencies=[Depends(mcp_auth_interceptor)]
                 ),
                 include_operations = ["analyze_code", "load_tests"]
                 )
mcp.mount()



if __name__ == "__main__":
    """Run the application with uvicorn."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.API_ENV == "development",
    )

