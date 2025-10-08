#!/bin/bash

# India Agricultural Intelligence Platform - Development Server
# Run the agricultural AI platform in development mode for testing

echo "🌿 INDIA AGRICULTURAL INTELLIGENCE PLATFORM - DEVELOPMENT MODE"
echo "================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Environment setup
print_step "Setting up development environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PORT=${PORT:-8000}
export RELOAD=true

# Check if we're in the right directory
if [ ! -f "india_agri_platform/api/main.py" ]; then
    echo -e "${RED}[ERROR]${NC} Not in the correct project directory."
    echo -e "${RED}[ERROR]${NC} Please run this script from the root of the Crop-Yield-Prediction project."
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p india_agri_platform/models

print_step "Starting development server with auto-reload..."

# Development server configuration display
echo ""
echo "🚀 DEVELOPMENT CONFIGURATION:"
echo "   Host: 127.0.0.1"
echo "   Port: $PORT"
echo "   Auto-reload: Enabled"
echo "   Workers: 1 (single process)"
echo ""
echo "🌐 Development Endpoints:"
echo "   API Docs: http://localhost:$PORT/api/docs"
echo "   Alternative Docs: http://localhost:$PORT/api/redoc"
echo "   Login to API docs for interactive testing"
echo ""
echo "🔄 The server will automatically reload when you make code changes."
echo "   Press Ctrl+C to stop the development server."
echo ""

# Start development server with auto-reload
python3 -m uvicorn \
    india_agri_platform.api.main:app \
    --host 127.0.0.1 \
    --port $PORT \
    --reload \
    --log-level info \
    --access-log

echo ""
print_status "Development server stopped."
echo ""
echo "📚 Next Steps:"
echo "  1. For production deployment: ./deploy/production.sh"
echo "  2. For testing predictions: curl http://localhost:8000/api/health"
echo "  3. For mobile app integration: Check API documentation at /api/docs"
