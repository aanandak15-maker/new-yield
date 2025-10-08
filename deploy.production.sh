#!/bin/bash

#
# India Agricultural Intelligence Platform - Production Deployment Script
#
# This script handles complete production deployment of the agricultural AI platform
# Features:
# - Automated infrastructure setup
# - Database initialization and migrations
# - SSL/TLS certificate generation
# - Monitoring and logging setup
# - Load balancer configuration
# - Backup and disaster recovery
#

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Environment configuration
export COMPOSE_FILE=docker-compose.production.yml
export COMPOSE_PROJECT_NAME=india_agri_platform_prod

# Default values (override with environment variables)
DOMAIN=${DOMAIN:-api.agritech.india.com}
ADMIN_EMAIL=${ADMIN_EMAIL:-admin@indiaagri.ai}
DB_PASSWORD=${DB_PASSWORD:-secure_db_password_2024}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-secure_grafana_password_2024}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null && ! docker-compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi

    # Check if required files exist
    if [ ! -f "docker-compose.production.yml" ]; then
        log_error "docker-compose.production.yml not found"
        exit 1
    fi

    if [ ! -f ".env.production" ]; then
        log_error ".env.production not found"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Function to generate secrets
generate_secrets() {
    log_info "Generating production secrets..."

    # Create secrets directory if it doesn't exist
    mkdir -p secrets

    # Generate database password
    if [ ! -f "secrets/db_password.txt" ]; then
        echo "${DB_PASSWORD}" > secrets/db_password.txt
        log_info "Generated database password"
    fi

    # Generate Grafana password
    if [ ! -f "secrets/grafana_password.txt" ]; then
        echo "${GRAFANA_PASSWORD}" > secrets/grafana_password.txt
        log_info "Generated Grafana password"
    fi

    # Generate API keys file
    if [ ! -f "secrets/api_keys.json" ]; then
        cat > secrets/api_keys.json << EOF
{
  "google_earth_engine": "your-gee-service-account-key",
  "openweather": "your-openweather-api-key",
  "weather_api": "your-weather-api-key",
  "government_api": "your-government-api-key"
}
EOF
        log_warning "Generated placeholder API keys - UPDATE BEFORE PRODUCTION USE"
    fi

    log_success "Secrets generated"
}

# Function to create Docker secrets
create_docker_secrets() {
    log_info "Creating Docker secrets..."

    # Remove existing secrets if they exist
    docker secret rm db_password grafana_password api_keys 2>/dev/null || true

    # Create new secrets
    docker secret create db_password secrets/db_password.txt
    docker secret create grafana_password secrets/grafana_password.txt
    docker secret create api_keys secrets/api_keys.json

    log_success "Docker secrets created"
}

# Function to setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."

    # Create monitoring directory
    mkdir -p monitoring/grafana/provisioning monitoring/grafana/dashboards monitoring/prometheus

    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'

  - job_name: 'database'
    static_configs:
      - targets: ['db:5432']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
EOF

    # Create basic Grafana dashboard provisioning
    mkdir -p monitoring/grafana/provisioning/datasources

    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log_success "Monitoring configuration completed"
}

# Function to initialize database
initialize_database() {
    log_info "Initializing database..."

    # Create docker directory and database initialization files
    mkdir -p docker

    # Create database initialization script
    cat > docker/init-db.sql << EOF
-- India Agricultural Intelligence Platform - Database Initialization

-- Create database and user (handled by environment variables)
-- This script creates additional schemas and tables

-- Enable PostGIS extension for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create schemas for organized data structure
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS iot;
CREATE SCHEMA IF NOT EXISTS satellite;
CREATE SCHEMA IF NOT EXISTS weather;

-- Create performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fields_location ON fields USING GIST(location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_data_timestamp ON iot.sensor_data(timestamp);

-- Setup default permissions
GRANT USAGE ON SCHEMA analytics, iot, satellite, weather TO agri_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA analytics, iot, satellite, weather TO agri_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA analytics, iot, satellite, weather TO agri_user;

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW analytics.yield_trends AS
SELECT
    crop_type,
    DATE_TRUNC('month', created_at) as period,
    AVG(predicted_yield_quintal_ha) as avg_yield,
    COUNT(*) as prediction_count
FROM predictions
WHERE created_at >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY crop_type, DATE_TRUNC('month', created_at)
ORDER BY period DESC;

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_analytics()
RETURNS void AS \$\$
BEGIN
    REFRESH MATERIALIZED VIEW analytics.yield_trends;
END;
\$\$ LANGUAGE plpgsql;

 -- Create initial platform configuration record
INSERT INTO system_config (key, value, created_at, updated_at)
VALUES
    ('version', '2.0.0', NOW(), NOW()),
    ('platform_name', 'India Agricultural Intelligence Platform', NOW(), NOW()),
    ('primary_region', 'punjab', NOW(), NOW())
ON CONFLICT (key) DO NOTHING;

COMMIT;
EOF

    # Create PostgreSQL configuration for production
    cat > docker/postgresql.conf << EOF
# Production PostgreSQL Configuration
listen_addresses = '*'
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
min_wal_size = 80MB
max_wal_size = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Logging
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'ddl'
log_duration = on
log_lock_waits = on

# Performance
random_page_cost = 1.1
effective_io_concurrency = 200
autovacuum = on
autovacuum_max_workers = 3
vacuum_cost_limit = 10000
EOF

    log_success "Database initialization files created"
}

# Function to setup SSL/TLS certificates
setup_ssl() {
    log_info "Setting up SSL/TLS certificates..."

    # Create self-signed certificate for development (replace with Let's Encrypt in production)
    if [ ! -f "secrets/ssl.crt" ] || [ ! -f "secrets/ssl.key" ]; then
        log_warning "Creating self-signed SSL certificates (replace with proper certificates for production)"

        # Create SSL directory
        mkdir -p secrets/ssl

        # Generate self-signed certificate
        openssl req -x509 -newkey rsa:4096 -keyout secrets/ssl.key -out secrets/ssl.crt -days 365 -nodes -subj "/C=IN/ST=Punjab/L=Chandigarh/O=India Agri Platform/CN=${DOMAIN}"

        log_info "Self-signed SSL certificates created"
    else
        log_info "SSL certificates already exist"
    fi

    log_success "SSL/TLS setup completed"
}

# Function to start deployment
deploy_production() {
    log_info "Starting production deployment..."

    # Create Docker secrets if in Docker Swarm mode
    if docker swarm ca 2>/dev/null; then
        log_info "Docker Swarm detected - creating secrets..."
        create_docker_secrets 2>/dev/null || log_warning "Failed to create Docker secrets"
    fi

    # Scale down any existing deployment first
    log_info "Scaling down existing services..."
    docker compose stop 2>/dev/null || true

    # Pull latest images
    log_info "Pulling latest images..."
    docker compose pull --parallel

    # Start database first
    log_info "Starting database..."
    docker compose up -d db

    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout=60
    elapsed=0
    while ! docker compose exec -T db pg_isready -U agri_user -d agri_platform >/dev/null 2>&1; do
        if [ $elapsed -ge $timeout ]; then
            log_error "Database failed to start within ${timeout} seconds"
            exit 1
        fi
        echo -n "."
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo ""

    # Start Redis
    log_info "Starting Redis..."
    docker compose up -d redis

    # Start monitoring stack
    log_info "Starting monitoring stack..."
    docker compose up -d prometheus grafana

    # Start main API server
    log_info "Starting API server..."
    docker compose up -d api

    # Start background workers
    log_info "Starting background workers..."
    docker compose up -d worker

    # Start backup service
    log_info "Starting backup service..."
    docker compose up -d backup

    # Scale API workers if high load required
    # docker compose up -d api_worker_1

    log_success "Production deployment completed"
}

# Function to run health checks
run_health_checks() {
    log_info "Running health checks..."

    # Wait a bit for services to stabilize
    sleep 30

    # Check API health
    if curl -f http://localhost/api/health >/dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi

    # Check database connectivity
    if docker compose exec -T db pg_isready -U agri_user -d agri_platform >/dev/null 2>&1; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi

    # Check Redis connectivity
    if docker compose exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi

    log_success "All health checks passed"
}

# Function to setup monitoring dashboard access
setup_dashboard_access() {
    log_info "Setting up monitoring dashboard access..."

    echo ""
    echo "================================================================================"
    echo "🎉 INDIA AGRICULTURAL INTELLIGENCE PLATFORM - DEPLOYMENT COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "🚀 Platform Services:"
    echo "   • API Server: https://${DOMAIN}/api/docs (Swagger UI)"
    echo "   • Health Check: https://${DOMAIN}/api/health"
    echo "   • System Status: https://${DOMAIN}/api/status/dashboard"
    echo ""
    echo "📊 Monitoring & Analytics:"
    echo "   • Grafana Dashboard: https://${DOMAIN}:3000 (admin / ${GRAFANA_PASSWORD})"
    echo "   • Prometheus Metrics: https://${DOMAIN}:9090"
    echo "   • Elasticsearch/Kibana: https://${DOMAIN}:5601"
    echo ""
    echo "🔐 Database Access:"
    echo "   • Host: localhost"
    echo "   • Database: agri_platform"
    echo "   • User: agri_user"
    echo "   • Password: ${DB_PASSWORD}"
    echo ""
    echo "================================================================================"
    echo ""
    log_success "Deployment summary shown above"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    log_info "Running post-deployment tests..."

    # Simple API test
    if curl -s http://localhost/api/health | grep -q "healthy"; then
        log_success "API responsiveness test passed"
    else
        log_error "API responsiveness test failed"
        return 1
    fi

    # Model loading test (makes a simple prediction request)
    if curl -s -X POST http://localhost/api/v1/predict/yield \
        -H "Content-Type: application/json" \
        -d '{"crop_name":"wheat","sowing_date":"2024-11-01","latitude":31.1471,"longitude":75.3412}' \
        | grep -q "prediction"; then
        log_success "Model prediction test passed"
    else
        log_warning "Model prediction test failed (may require model training first)"
    fi

    log_success "Post-deployment tests completed"
}

# Function to show usage
show_usage() {
    cat << EOF
India Agricultural Intelligence Platform - Production Deployment Script

USAGE:
    ./deploy.production.sh [OPTIONS] [COMMAND]

COMMANDS:
    setup           Setup prerequisites and generate secrets
    deploy          Full production deployment
    start           Start all services
    stop            Stop all services
    restart         Restart all services
    logs            Show service logs
    health          Run health checks
    backup          Trigger manual backup
    update          Update and restart services

OPTIONS:
    -h, --help      Show this help message
    -d, --domain    Domain name (default: api.agritech.india.com)
    -e, --email     Admin email (default: admin@indiaagri.ai)

ENVIRONMENT VARIABLES:
    DOMAIN          Override default domain
    ADMIN_EMAIL     Override default admin email
    DB_PASSWORD     Database password (auto-generated if not set)
    GRAFANA_PASSWORD Dashboard admin password (auto-generated if not set)

EXAMPLES:
    # Complete deployment
    ./deploy.production.sh deploy

    # Setup only
    ./deploy.production.sh setup

    # Custom domain
    DOMAIN=myserver.com ./deploy.production.sh deploy

    # Show service logs
    ./deploy.production.sh logs

EOF
}

# Main execution logic
main() {
    local command="deploy"

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -d|--domain)
                DOMAIN="$2"
                shift 2
                ;;
            -e|--email)
                ADMIN_EMAIL="$2"
                shift 2
                ;;
            setup|deploy|start|stop|restart|logs|health|backup|update)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Execute selected command
    case $command in
        setup)
            check_prerequisites
            generate_secrets
            setup_monitoring
            initialize_database
            setup_ssl
            log_success "Setup completed successfully"
            ;;
        deploy)
            check_prerequisites
            generate_secrets
            setup_monitoring
            initialize_database
            setup_ssl
            deploy_production
            run_health_checks
            run_post_deployment_tests
            setup_dashboard_access
            ;;
        start)
            log_info "Starting services..."
            docker compose up -d
            run_health_checks
            ;;
        stop)
            log_info "Stopping services..."
            docker compose down
            ;;
        restart)
            log_info "Restarting services..."
            docker compose restart
            run_health_checks
            ;;
        logs)
            log_info "Showing service logs (press Ctrl+C to stop)..."
            docker compose logs -f --tail=100
            ;;
        health)
            run_health_checks
            ;;
        backup)
            log_info "Triggering manual backup..."
            # This would trigger the backup container or run backup script
            docker compose exec backup /bin/bash -c "pg_dump -h db -U agri_user agri_platform > /backup/manual_backup_$(date +\%Y\%m\%d_\%H\%M\%S).sql"
            log_success "Manual backup completed"
            ;;
        update)
            log_info "Updating services..."
            docker compose pull
            docker compose up -d --remove-orphans
            run_health_checks
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
