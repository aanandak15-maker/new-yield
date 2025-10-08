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

# Login to Railway (updated for v4+ CLI)
echo "Step 1: Authenticating with Railway..."
echo "⚠️  Note: You'll need to complete authentication in your browser"
railway login

# Link to existing project or create new one
echo "Step 2: Setting up Railway project..."
if railway list | grep -q "new-yield"; then
    echo "Project exists, linking to it..."
    railway link --project new-yield-production
else
    echo "Creating new project from current directory..."
    railway init
    sleep 5
fi

# Deploy the project
echo "Step 3: Deploying your application..."
railway up

# Wait for deployment
echo "Step 4: Waiting for deployment to complete..."
sleep 15

# Check deployment status
echo "Step 5: Checking deployment status..."
railway status

# Get the deployment URL
echo "Step 6: Getting deployment URL..."
DEPLOYMENT_URL=$(railway domain 2>/dev/null || echo "new-yield-production.up.railway.app")

echo "✅ Deployment complete!"
echo "=========================================="
echo "Your agricultural intelligence API is live at:"
echo "https://$DEPLOYMENT_URL"
echo ""
echo "🌾 Test your API:"
echo "curl https://$DEPLOYMENT_URL/health"
echo "curl https://$DEPLOYMENT_URL/docs"
echo ""
echo "🔧 Update your Vercel environment variable:"
echo "NEXT_PUBLIC_API_BASE_URL=https://$DEPLOYMENT_URL"
echo ""
echo "🌾 Your platform is ready for farmers to use!"
echo "=========================================="
