flowchart TD
    %% API Request Flow

    subgraph "📱 FARMER APP"
        FE[📱 Farmer Interface<br/>Web/Mobile App]
        AUTH_TOKEN[JWT Token<br/>User Authentication]
        GPS_LOCATION[📍 GPS Coordinates<br/>Real-time location]
        USER_INPUT[🌾 Crop, Variety, Season<br/>Farmer preferences]
    end

    subgraph "🌐 FASTAPI GATEWAY"
        CORS[🔒 CORS Validation<br/>Cross-origin checks]
        AUTH[🔐 Authentication<br/>JWT verification]
        RATE_LIMIT[⚡ Rate Limiting<br/>10 req/min free tier]
        LOGGING[📝 Request Logging<br/>Audit trail]
    end

    subgraph "🎯 API ROUTE HANDLING"
        ROUTER[🏗️ Route Matching<br/>25+ endpoint routing]
        VALIDATE[✅ Input Validation<br/>Pydantic models]
        FORMAT[🔧 Data Formatting<br/>Request preprocessing]
    end

    subgraph "🧠 UNIFIED PREDICTOR ENGINE"
        COORD_PROC[📍 Coordinate Processing<br/>State/district detection]
        CROP_DETECT[🌱 Crop Auto-Detection<br/>Season/location matching]
        MODEL_SELECT[🎯 Model Selection<br/>Crop-state matching]
        FEATURE_ENG[⚙️ Feature Engineering<br/>Real-time data integration]
    end

    subgraph "🤖 ML MODEL INFERENCE"
        ENSEMBLE[🎭 Ensemble Voting<br/>Multiple model integration]
        CONFIDENCE[📊 Confidence Calculation<br/>Uncertainty estimation]
        CALIBRATE[⚖️ Prediction Calibration<br/>Range estimation]
    end

    subgraph "🔄 REAL-TIME DATA FETCH"
        WEATHER[🌤️ OpenWeather API<br/>Current + forecast weather]
        SATELLITE[🛰️ Google Earth Engine<br/>NDVI, soil moisture]
        MARKET[💰 Commodity Market APIs<br/>Live prices, MSP]
        GOVT[🏛️ Government APIs<br/>Subsidy schemes, MSP]
    end

    subgraph "💼 BUSINESS LOGIC PROCESSING"
        REVENUE[💵 Revenue Calculation<br/>Market price * yield]
        COST_CALC[💸 Cost Analysis<br/>Input costs, labor]
        PROFIT_MARG[📈 Profit Margin<br/>Revenue - costs]
        RISK_ASSESS[🎯 Risk Assessment<br/>Weather, price, pest risk]
    end

    subgraph "🧬 AI ADVISORY SYSTEM"
        GEMINI[🎭 Gemini AI Consultant<br/>Expert advice generator]
        ETHICAL[⚖️ Ethical Orchestrator<br/>Organic-first guidelines]
        CONTEXT[🧠 Contextual Reasoning<br/>Farmer situation awareness]
        RESPONSE[💬 Natural Language Response<br/>Local language output]
    end

    subgraph "📊 OUTPUT FORMATTING"
        JSON_FORMAT[📄 JSON Response<br/>Structured API output]
        CHART_DATA[📈 Chart Visualization<br/>Dashboard-ready data]
        RECOMMEND[📋 Action Recommendations<br/>Priority-labeled tasks]
        ALERTS[🚨 Weather/Pest Alerts<br/>Time-sensitive warnings]
    end

    subgraph "🔄 RESPONSE OPTIMIZATION"
        CACHE[⚡ Redis Cache<br/>Prediction caching]
        COMPRESS[🗜️ Response Compression<br/>Mobile optimization]
        METRICS[📊 Performance Metrics<br/>Response time tracking]
    end

    %% Request Flow
    FE --> CORS
    CORS --> AUTH
    AUTH --> RATE_LIMIT
    RATE_LIMIT --> LOGGING
    LOGGING --> ROUTER
    ROUTER --> VALIDATE
    VALIDATE --> FORMAT
    FORMAT --> COORD_PROC

    COORD_PROC --> CROP_DETECT
    CROP_DETECT --> MODEL_SELECT
    MODEL_SELECT --> FEATURE_ENG

    FEATURE_ENG --> WEATHER
    FEATURE_ENG --> SATELLITE
    FEATURE_ENG --> MARKET
    FEATURE_ENG --> GOVT

    WEATHER --> ENSEMBLE
    SATELLITE --> ENSEMBLE
    MARKET --> ENSEMBLE
    GOVT --> ENSEMBLE

    ENSEMBLE --> CONFIDENCE
    CONFIDENCE --> CALIBRATE

    CALIBRATE --> REVENUE
    CALIBRATE --> COST_CALC
    COST_CALC --> PROFIT_MARG
    PROFIT_MARG --> RISK_ASSESS

    RISK_ASSESS --> GEMINI
    GEMINI --> ETHICAL
    ETHICAL --> CONTEXT
    CONTEXT --> RESPONSE

    CALIBRATE --> JSON_FORMAT
    REVENUE --> JSON_FORMAT
    PROFIT_MARG --> JSON_FORMAT
    RISK_ASSESS --> JSON_FORMAT
    RESPONSE --> JSON_FORMAT

    RESPONSE --> CHART_DATA
    RESPONSE --> RECOMMEND
    RESPONSE --> ALERTS

    JSON_FORMAT --> CACHE
    CACHE --> COMPRESS
    COMPRESS --> METRICS

    classDef client fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef gateway fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef routing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef processing fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef ml fill:#fff8e1,stroke:#ff9800,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef business fill:#f1f8e9,stroke:#4caf50,stroke-width:2px
    classDef ai fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef output fill:#f8f9fa,stroke:#607d8b,stroke-width:2px
    classDef optimization fill:#fff8f7,stroke:#ff5722,stroke-width:2px

    class FE,AUTH_TOKEN,GPS_LOCATION,USER_INPUT client
    class CORS,AUTH,RATE_LIMIT,LOGGING gateway
    class ROUTER,VALIDATE,FORMAT routing
    class COORD_PROC,CROP_DETECT,MODEL_SELECT,FEATURE_ENG processing
    class ENSEMBLE,CONFIDENCE,CALIBRATE ml
    class WEATHER,SATELLITE,MARKET,GOVT external
    class REVENUE,COST_CALC,PROFIT_MARG,RISK_ASSESS business
    class GEMINI,ETHICAL,CONTEXT,RESPONSE ai
    class JSON_FORMAT,CHART_DATA,RECOMMEND,ALERTS output
    class CACHE,COMPRESS,METRICS optimization
