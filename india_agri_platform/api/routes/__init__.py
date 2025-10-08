"""
India Agricultural Intelligence Platform - API Routes

FastAPI route modules for all platform services
"""

# Create routers - prevent circular imports
from fastapi import APIRouter

yield_router = APIRouter()
analytics_router = APIRouter()
satellite_router = APIRouter()
weather_router = APIRouter()
iot_router = APIRouter()
external_router = APIRouter()
dashboard_router = APIRouter()

# Export routers
__all__ = [
    'yield_router',
    'analytics_router',
    'satellite_router',
    'weather_router',
    'iot_router',
    'external_router',
    'dashboard_router'
]
