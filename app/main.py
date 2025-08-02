import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.config import settings
from app.api.routes import router
from app.utils.helpers import optimize_memory
from typing import AsyncGenerator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# The lifespan context manager now handles startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown events.
    """
    logger.info("Starting LLM Document Query System...")
    optimize_memory()
    yield  # The application runs here
    logger.info("Shutting down LLM Document Query System...")
    optimize_memory()

app = FastAPI(
    title=settings.app_name,
    description="Optimized LLM Document Query System",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan  # Use the lifespan context manager
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Include API routes
app.include_router(router, prefix="/api/v1")

# The root endpoint to serve the frontend
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
