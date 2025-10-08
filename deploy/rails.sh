#!/bin/bash

# Railway Production Deployment Script
# This script sets up the agricultural intelligence platform on Railway

echo "🚀 Deploying Agricultural Intelligence Platform to Railway..."
echo "=================================================="

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
echo "Step 1: Authenticating with Railway..."
railway login --yes

# Create new Railway project
echo "Step 2: Creating Railway project..."
railway init new-yield --yes
railway up

# Wait for deployment
echo "Step 3: Waiting for deployment..."
sleep 10

# Check deployment status
echo "Step 4: Checking deployment status..."
railway status

# Get the deployment URL
echo "Step 5: Getting deployment URL..."
DEPLOYMENT_URL=$(railway domain)

echo "✅ Deployment complete!"
echo "=========================================="
echo "Your agricultural intelligence API is live at:"
echo "https://$DEPLOYMENT_URL"
echo ""
echo "🔧 Update your Vercel environment variable:"
echo "NEXT_PUBLIC_API_BASE_URL=https://$DEPLOYMENT_URL"
echo ""
echo "🌾 Your platform is ready for farmers to use!"
echo "=========================================="
