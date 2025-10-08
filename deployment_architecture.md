flowchart TB
    subgraph "🌐 GLOBAL EDGE NETWORK"
        CDN[CDN Distribution<br/>Global edge nodes<br/>Content delivery]
        WAF[Web Application Firewall<br/>DDoS protection<br/>Traffic filtering]
    end

    subgraph "🚀 APPLICATION LOAD BALANCER"
        ALB[ALB/Nginx<br/>Auto-scaling<br/>Session persistence]
        TCP_CHECK[Health Checks<br/>Node monitoring<br/>Failover routing]
    end

    subgraph "⚙️ AUTO-SCALING CLUSTER"
        ASG[Auto Scaling Groups<br/>EC2 instances<br/>Kubernetes pods]
        SPOT_INSTANCES[Spot Instances<br/>Cost optimization<br/>80% savings]
        RESERVED_INSTANCES[Reserved Instances<br/>Production workload<br/>20-50% savings]
    end

    subgraph "🐳 CONTAINER ORCHESTRATION"
        K8S_MASTER[Kubernetes Master<br/>API server<br/>Scheduler controller]
        ETCD[(etcd Cluster<br/>Configuration store<br/>Service discovery)]
        WORKER_NODES[Worker Nodes<br/>Container runtime<br/>Application pods]
    end

    subgraph "📦 CONTAINER REGISTRY"
        ECR[Docker Registry<br/>Image versioning<br/>Security scanning]
        ARTIFACTS[Build Artifacts<br/>Deployment packages<br/>System images]
    end

    subgraph "🏗️ APPLICATION SERVERS"
        FASTAPI[FastAPI Application<br/>25+ REST endpoints<br/>Async processing]
        GUNICORN[Gunicorn Workers<br/>4 workers/instance<br/>Process management]
        UVICORN[Uvicorn ASGI<br/>Concurrent requests<br/>WebSocket support]
    end

    subgraph "🗄️ DATABASE LAYER"
        POSTGRES_MASTER[(PostgreSQL Master<br/>Write operations<br/>Replication source)]
        POSTGRES_REPLICAS[(PostgreSQL Read Replicas<br/>3x horizontal scaling<br/>Query optimization)]
        CONNECTION_POOL[Connection Pooler<br/>PgBouncer<br/>Resource optimization]
    end

    subgraph "💾 CACHE & QUEUE LAYER"
        REDIS_CACHE{{Redis Cluster<br/>In-memory cache<br/>Session management}}
        REDIS_QUEUE{{Redis Queue<br/>Background jobs<br/>Task scheduling}}
        REDIS_PUBSUB{{Redis Pub/Sub<br/>Real-time notifications<br/>Event streaming}}
    end

    subgraph "🔐 IDENTITY & AUTH"
        FIREBASE_AUTH[Firebase Auth<br/>JWT tokens<br/>User management]
        CUSTOM_AUTH[Custom OAuth2<br/>Role-based access<br/>Multi-tenant support]
        JWT_VALIDATOR[JWT Token Validation<br/>Refresh token handling<br/>Session management]
    end

    subgraph "📊 ANALYTICS & MONITORING"
        PROMETHEUS[Prometheus Metrics<br/>Time-series data<br/>Alerting rules]
        GRAFANA[Grafana Dashboards<br/>Performance metrics<br/>Business KPIs]
        ELASTICSEARCH[(Elasticsearch Cluster<br/>Log aggregation<br/>Full-text search)]
    end

    subgraph "☁️ INFRASTRUCTURE AS CODE"
        TERRAFORM[Terraform Workspaces<br/>Environment provisioning<br/>Configuration management]
        ANSIBLE[Ansible Playbooks<br/>Deployment automation<br/>Configuration updates]
        GITOPS[GitOps Workflows<br/>Automated deployments<br/>Version control]
    end

    subgraph "🔒 SECURITY & COMPLIANCE"
        VAULT[HashiCorp Vault<br/>Secret management<br/>Encryption keys]
        AWS_KMS[AWS KMS<br/>Data encryption<br/>Key rotation]
        SSL_TERMINATION[SSL Termination<br/>Certificate management<br/>HTTPS enforcement]
    end

    subgraph "📡 EXTERNAL INTEGRATIONS"
        OPENWEATHER[OpenWeather API<br/>Live weather data<br/>Geocoding service]
        GOOGLE_EARTH[Google Earth Engine<br/>Satellite imagery<br/>NDVI analysis]
        GOVT_APIS[Government APIs<br/>MSP data access<br/>Subsidy schemes]
        MARKET_DATA[Market Data Providers<br/>Commodity prices<br/>MSP information]
    end

    %% Traffic Flow
    CDN --> WAF
    WAF --> ALB
    ALB --> ASG
    ASG --> K8S_MASTER
    K8S_MASTER --> WORKER_NODES
    WORKER_NODES --> ECR

    WORKER_NODES --> FASTAPI
    FASTAPI --> GUNICORN
    GUNICORN --> UVICORN

    UVICORN --> POSTGRES_MASTER
    UVICORN --> POSTGRES_REPLICAS
    UVICORN --> CONNECTION_POOL

    UVICORN --> REDIS_CACHE
    UVICORN --> REDIS_QUEUE
    UVICORN --> REDIS_PUBSUB

    UVICORN --> FIREBASE_AUTH
    FIREBASE_AUTH --> CUSTOM_AUTH
    CUSTOM_AUTH --> JWT_VALIDATOR

    POSTGRES_MASTER --> POSTGRES_REPLICAS
    POSTGRES_REPLICAS --> CONNECTION_POOL

    REDIS_CACHE --> CONNECTION_POOL
    REDIS_QUEUE --> CONNECTION_POOL
    REDIS_PUBSUB --> CONNECTION_POOL

    ASG --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    PROMETHEUS --> ELASTICSEARCH

    TERRAFORM --> ASG
    ANSIBLE --> APPLICATION_SERVERS
    GITOPS --> ASG

    VAULT --> FIREBASE_AUTH
    AWS_KMS --> FIREBASE_AUTH
    SSL_TERMINATION --> ALB

    UVICORN --> OPENWEATHER
    UVICORN --> GOOGLE_EARTH
    UVICORN --> GOVT_APIS
    UVICORN --> MARKET_DATA

    classDef network fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef load_balance fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef scaling fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef container fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef registry fill:#fff8e1,stroke:#ff9800,stroke-width:2px
    classDef app fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef database fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef cache fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    classDef auth fill:#fff8f7,stroke:#ff5722,stroke-width:2px
    classDef monitoring fill:#f8f9fa,stroke:#607d8b,stroke-width:2px
    classDef infra fill:#e9ecef,stroke:#6c757d,stroke-width:2px
    classDef security fill:#f5c6cb,stroke:#dc3545,stroke-width:2px
    classDef external fill:#d4edda,stroke:#198754,stroke-width:2px

    class CDN,WAF network
    class ALB,TCP_CHECK load_balance
    class ASG,SPOT_INSTANCES,RESERVED_INSTANCES scaling
    class K8S_MASTER,ETCD,WORKER_NODES container
    class ECR,ARTIFACTS registry
    class FASTAPI,GUNICORN,UVICORN app
    class POSTGRES_MASTER,POSTGRES_REPLICAS,CONNECTION_POOL database
    class REDIS_CACHE,REDIS_QUEUE,REDIS_PUBSUB cache
    class FIREBASE_AUTH,CUSTOM_AUTH,JWT_VALIDATOR auth
    class PROMETHEUS,GRAFANA,ELASTICSEARCH monitoring
    class TERRAFORM,ANSIBLE,GITOPS infra
    class VAULT,AWS_KMS,SSL_TERMINATION security
    class OPENWEATHER,GOOGLE_EARTH,GOVT_APIS,MARKET_DATA external
