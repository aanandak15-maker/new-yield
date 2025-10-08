```mermaid
flowchart TB
    %% Platform Architecture Overview

    subgraph "🌾 AGRICULTURAL INTELLIGENCE PLATFORM"
        subgraph "👨‍🌾 FARMER INTERACTION LAYER"
            FE[📱 Mobile/Web Interface<br/>Farmer Dashboard]
            GPS[📍 GPS Location<br/>Auto-Detection]
            VO[🎙️ Voice Commands<br/>Multi-Language Support]
        end

        subgraph "🤖 AI AGRICULTURAL BRAIN"
            UP[🧠 Unified Predictor Engine<br/>Multi-Crop Intelligence]
            subgraph "Crop Ecosystems"
                Rice[🌾 Rice Models<br/>10 States]
                Wheat[🌾 Wheat Models<br/>8 States]
                Cotton[🌿 Cotton Models<br/>10 States]
                Maize[🌽 Maize Models<br/>8 States]
            end
            OI[🧬 Gemini AI Consultant<br/>Ethical Orchestrator]
        end

        subgraph "🌐 API & MICRO-SERVICES"
            API[🚀 FastAPI<br/>25+ Endpoints]
            AUTH[🔐 Firebase Auth<br/>Farmer Authentication]
            DB[(💾 PostgreSQL<br/>Agricultural Data Store)]
            CACHE[⚡ Redis Cache<br/>Real-time Performance]
        end

        subgraph "📡 REAL-TIME DATA INTEGRATION"
            OW[🌤️ OpenWeather API<br/>Live Weather Data]
            GEE[🛰️ Google Earth Engine<br/>Satellite Analytics]
            GOV[🏛️ Government APIs<br/>MSP & Subsidy Data]
            SOILINT[🌱 Soil Intelligence<br/>IoT Sensor Networks]
        end

        subgraph "📊 BUSINESS & ANALYTICS"
            BM[🤑 Business Model Engine<br/>Freemium → Premium]
            PART[🤝 Partnership Pipeline<br/>Government + Cooperatives]
            BMET[📈 Revenue Analytics<br/>Farm-level Insights]
            REP[📋 AI Reports<br/>Digital Dashboards]
        end

        subgraph "🏗️ INFRASTRUCTURE LAYER"
            DC[🐳 Docker<br/>Container Orchestration]
            CLOUD[☁️ Cloud Hosting<br/>Auto-scaling]
            LOG[📝 Logging & Monitoring<br/>Health Dashboard]
            SEC[🔒 Security Layer<br/>Data Encryption]
        end

        subgraph "🎯 SCALING & PARTNERSHIPS"
            GOV[🏛️ Government<br/>State + Central]
            COOP[🏪 Cooperatives<br/>IFFCO, NAFED, Amul]
            INPUT[🏭 Input Companies<br/>Bayer, UPL, Coromandel]
            TECH[⚙️ Technology<br/>Jio, Airtel, Mahindra]
        end
    end

    %% Data Flow Connections
    FE --> GPS
    GPS --> UP
    UP --> Rice
    UP --> Wheat
    UP --> Cotton
    UP --> Maize

    FE --> API
    API --> AUTH
    AUTH --> DB
    API --> CACHE

    UP --> OI
    OI --> OW
    OI --> GEE
    OI --> GOV
    OI --> SOILINT

    UP --> BMET
    BMET --> REP
    REP --> FE

    BM --> PART
    PART --> GOV
    PART --> COOP
    PART --> INPUT
    PART --> TECH

    API --> DC
    DC --> CLOUD
    CLOUD --> LOG
    LOG --> SEC

    classDef frontend fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef ai fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef business fill:#fff3e0,stroke:#f57f17,stroke-width:2px
    classDef infrastructure fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef partners fill:#f1f8e9,stroke:#558b2f,stroke-width:2px

    class FE,GPS,VO frontend
    class UP,Rice,Wheat,Cotton,Maize,OI ai
    class API,AUTH,DB,CACHE data
    class BM,PART,BMET,REP business
    class DC,CLOUD,LOG,SEC infrastructure
    class GOV,COOP,INPUT,TECH partners

    OW:::data
    GEE:::data
