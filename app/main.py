"""
FastAPI application entry point for agent MCP server
"""

from contextlib import asynccontextmanager
import uvicorn
from tortoise import Tortoise
from tortoise import connections
import asyncio
from fastapi import FastAPI, Depends
from fastapi_mcp import FastApiMCP, AuthConfig
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings, TORTOISE_ORM
from app.exceptions.register import register_exception_handlers
from app.infrastructure.request_recorder.middleware import RecordRequestMiddleware
from app.logging_config import setup_logging
from app.routes import router as api_router
from app.infrastructure.queue_consumer import QueueConsumer
from app.config import supabase_queue
from app.utils.auth import mcp_auth_interceptor

logger = setup_logging()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper async signal handling"""
    global worker_service

    # Startup
    logger.info(f"Starting DevDox AI Agent Worker Service v{settings.VERSION}")

    try:
            logger.info("Database connection")

            # Initialize database
            if TORTOISE_ORM:
                await Tortoise.init(config=TORTOISE_ORM)
                logger.info("Database initialized")
                conn = Tortoise.get_connection("default")
                # Test query
                result = await conn.execute_query("SELECT 1 as test")
                logger.info(f"✅ Database connection verified: {result}")




            logger.info("Database initialized and connection verified")
            logger.info("Tortoise._inited: " + str(Tortoise._inited))

    except Exception as e:
            logger.error(f"Database connection  failed: {e}")


    worker_service = QueueConsumer(
        queue=supabase_queue,
        workers=getattr(settings, 'CONSUMER_WORKERS', 2),
        poll_interval=getattr(settings, 'CONSUMER_POLL_INTERVAL', 7.0),
        max_processing_time=getattr(settings, 'CONSUMER_MAX_PROCESSING_TIME', 1800)
    )

    _ = asyncio.create_task(worker_service.start())
    logger.info("Consumer started as background task")
    await asyncio.sleep(1)
    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Application stop initiated")
    if worker_service and worker_service.running:
        logger.info("Stopping consumer...")
        await worker_service.stop()
        logger.info("Consumer stopped")



    if TORTOISE_ORM:
        await Tortoise.close_connections()
        logger.info("Database connections closed")

    logger.info("Application shutdown complete")


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

app.add_middleware(
    RecordRequestMiddleware,
    include_rules=[
            {"operation_id": "analyze_code", "redact_keys": set()},
            {"operation_id": "load_tests",    "redact_keys": set()},
            {"operation_id": "qna_summary", "redact_keys": set()},
            # add one per included operation_id
        ],
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
                 include_operations = ["analyze_code", "load_tests", "qna_summary"]
                 )
mcp.mount()
# Streamable HTTP (for future-proofing)
mcp.mount_http(api_router, mount_path="/my-http")



if __name__ == "__main__":
    """Run the application with uvicorn."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.API_ENV == "development",
    )

