```mermaid
flowchart LR
    %% Data Processing Pipeline

    subgraph "📊 RAW DATA SOURCES"
        GOV[Govt. APY Data<br/>1997-2019<br/>1481 records]
        MLDATA[ML Training Data<br/>agriyield-2025.zip<br/>10,000 records]
        SPATIAL[Spatial Data<br/>archive.zip<br/>1625 records]
        REALTIME[Real-time APIs<br/>OpenWeather, GEE<br/>Live streaming]
        FARMER[Farmer Input<br/>GPS, preferences<br/>User-generated]
    end

    subgraph "🔄 DATA INGESTION LAYER"
        LOAD[Data Loaders<br/>APYLoader, MLLoader<br/>Format detection]
        VALIDATE[Data Validation<br/>Schema validation<br/>Missing value checks]
        TRANSFORM[Data Transformation<br/>Feature engineering<br/>Unit standardization]
        CLEANSE[Data Cleansing<br/>Outlier removal<br/>Duplicate handling]
    end

    subgraph "🤖 FEATURE ENGINEERING PIPELINE"
        SCALE[Feature Scaling<br/>StandardScaler<br/>MinMaxScaler]
        ENCODE[Categorical Encoding<br/>OneHot, Label encoding<br/>Target encoding]
        INTERACT[Feature Interactions<br/>Polynomial features<br/>Cross terms]
        SELECT[Feature Selection<br/>Correlation analysis<br/>Importance ranking]
    end

    subgraph "🎯 MODEL TRAINING WORKFLOW"
        SPLIT[Train/Val/Test Split<br/>80/10/10 ratio<br/>Stratified sampling]
        TRAIN[Model Training<br/>Random Forest, XGBoost<br/>Hyperparameter tuning]
        VALIDATE_TRAIN[Cross Validation<br/>5-fold CV<br/>Performance metrics]
        CALIBRATE[Model Calibration<br/>Probability calibration<br/>Confidence intervals]
    end

    subgraph "🔮 PREDICTION ENCODER"
        REAL_TIME[Real-time Data Fetch<br/>Weather, satellite<br/>API calls]
        FEATURE_PROC[Feature Processing<br/>Coordinate encoding<br/>Temporal features]
        MODEL_SELECT[Model Selection<br/>Crop-state matching<br/>Ensemble voting]
        CONFIDENCE[Confidence Calculation<br/>Uncertainty estimation<br/>Risk assessment]
    end

    subgraph "📋 OUTPUT FORMATTING"
        YIELD_PRED[Yield Prediction<br/>q/ha with range<br/>Growth stage]
        BUSINESS_METRICS[Revenue Calculation<br/>Market prices<br/>Cost analysis]
        RECOMMENDATIONS[Actionable Insights<br/>Crop rotation<br/>Pest management]
        REPORTS[Analytics Dashboard<br/>Performance charts<br/>Progress tracking]
    end

    subgraph "💾 DATA STORAGE & CACHING"
        DB[(PostgreSQL<br/>Historical data<br/>User profiles)]
        CACHE{{Redis Cache<br/>Prediction cache<br/>API responses}}
        BLOB[(Cloud Storage<br/>Model artifacts<br/>Large datasets)]
    end

    %% Data Flow Connections
    GOV --> LOAD
    MLDATA --> LOAD
    SPATIAL --> LOAD
    REALTIME --> LOAD
    FARMER --> LOAD

    LOAD --> VALIDATE
    VALIDATE --> TRANSFORM
    TRANSFORM --> CLEANSE
    CLEANSE --> SCALE
    SCALE --> ENCODE
    ENCODE --> INTERACT
    INTERACT --> SELECT

    SELECT --> SPLIT
    SPLIT --> TRAIN
    TRAIN --> VALIDATE_TRAIN
    VALIDATE_TRAIN --> CALIBRATE

    REAL_TIME --> FEATURE_PROC
    FEATURE_PROC --> MODEL_SELECT
    MODEL_SELECT --> CONFIDENCE
    CONFIDENCE --> CALIBRATE

    CALIBRATE --> YIELD_PRED
    YIELD_PRED --> BUSINESS_METRICS
    BUSINESS_METRICS --> RECOMMENDATIONS
    RECOMMENDATIONS --> REPORTS

    CLEANSE --> DB
    REPORTS --> CACHE
    CALIBRATE --> BLOB

    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef ml fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    class GOV,MLDATA,SPATIAL,REALTIME,FARMER input
    class LOAD,VALIDATE,TRANSFORM,CLEANSE processing
    class SCALE,ENCODE,INTERACT,SELECT,REAL_TIME,FEATURE_PROC,MODEL_SELECT,CONFIDENCE processing
    class SPLIT,TRAIN,VALIDATE_TRAIN,CALIBRATE ml
    class YIELD_PRED,BUSINESS_METRICS,RECOMMENDATIONS,REPORTS output
    class DB,CACHE,BLOB storage
