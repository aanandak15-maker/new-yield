#!/bin/bash

# India Agricultural Intelligence Platform - Production Deployment Script
# Deploy the agricultural AI platform for farmer applications and cooperatives

echo "🌾 INDIA AGRICULTURAL INTELLIGENCE PLATFORM - PRODUCTION LAUNCH"
echo "===================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "india_agri_platform/api/main.py" ]; then
    print_error "Not in the correct project directory."
    print_error "Please run this script from the root of the Crop-Yield-Prediction project."
    exit 1
fi

# Environment setup
print_step "Setting up production environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}
export ENV=production

# Create necessary directories
mkdir -p logs
mkdir -p india_agri_platform/models

# Check Python environment
print_step "Checking Python environment..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $python_version"

# Check if required packages are installed
print_step "Checking dependencies..."
pip install -q fastapi uvicorn python-multipart pydantic requests pandas numpy scikit-learn joblib matplotlib seaborn

# Check if models directory has content
if [ -d "india_agri_platform/models/cotton_models" ]; then
    model_count=$(find india_agri_platform/models/cotton_models -name "*.pkl" | wc -l)
    print_status "Found $model_count trained cotton models"
else
    print_warning "No trained models found - platform will run in educational mode"
fi

# Production server configuration
print_step "Starting production server..."

# Display deployment information
echo ""
echo "🚀 PRODUCTION DEPLOYMENT CONFIGURATION:"
echo "   Host: 0.0.0.0"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "   Environment: Production"
echo ""
echo "🌐 API Endpoints will be available at:"
echo "   Main API: http://localhost:$PORT/"
echo "   Health Check: http://localhost:$PORT/api/health"
echo "   API Docs: http://localhost:$PORT/api/docs"
echo "   Farmer Predictions: http://localhost:$PORT/api/v1/predict/yield"
echo ""

# Start the production server
print_status "Launching agricultural intelligence platform..."
echo "Press Ctrl+C to stop the server"
echo ""

# Use uvicorn for production deployment
python3 -m uvicorn \
    india_agri_platform.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers $WORKERS \
    --loop uvloop \
    --http httptools \
    --access-log \
    --log-level info \
    --proxy-headers \
    --forwarded-allow-ips "*"

# If the server stops, show next steps
echo ""
print_warning "Production server stopped."
echo ""
echo "🔄 Next Steps:"
echo "  1. For development mode: ./deploy/development.sh"
echo "  2. For Docker deployment: docker build -t agri-ai . && docker run -p 8000:8000 agri-ai"
echo "  3. For farmers' apps: Integrate API endpoints in your mobile/web applications"
echo "  4. For cooperatives: Contact for enterprise integration frameworks"
echo ""
echo "📞 Partnership Inquiries:"
echo "     Cotton cooperatives: Contact for BT cotton optimization modules"
echo "     Processing industries: Integration with starch/processing facilities"
echo "     Agricultural ministry: Policy planning and extension services"
echo ""
