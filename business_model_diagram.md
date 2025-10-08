flowchart LR
    %% Business Model Ecosystem

    subgraph "👨‍🌾 CUSTOMER SEGMENTS"
        SMALLHOLDER[Smallholder Farmers<br/>2-5 acres<br/>₹50K annual income]
        PROGRESSIVE[Progressive Farmers<br/>5-20 acres<br/>₹2-5 lakh income]
        COOPERATIVE[Farmer Cooperatives<br/>100-10,000 members<br/>₹10-500 crore turnover]
        AGRI_CORP[Agri-Business Corporations<br/>Input suppliers, processors<br/>₹100-1000 crore revenue]
    end

    subgraph "💰 VALUE PROPOSITIONS"
        FREEMIUM_VAL[Free Basic Intelligence<br/>Crop recommendations<br/>Weather insights<br/>Basic analytics]
        PREMIUM_VAL[Expert AI Consultations<br/>Unlimited predictions<br/>Market intelligence<br/>Precision farming]
        ENTERPRISE_VAL[Bulk Intelligence Solutions<br/>Cooperative management<br/>Supply chain optimization<br/>Government reporting]
        PARTNER_VAL[Data Monetization<br/>Aggregated insights<br/>Market intelligence<br/>Co-marketing]
    end

    subgraph "💳 REVENUE STREAMS"
        SUBSCRIPTION[Monthly Subscriptions<br/>Premium: ₹500/month<br/>Enterprise: ₹5000/month<br/>Annual: ₹15 crores]
        TRANSACTIONS[Transaction Fees<br/>Input supply: 2-5%<br/>MSP marketing: 1-3%<br/>Insurance: 0.5-1%]
        DATA_LICENSING[Data Licensing<br/>Government APIs: ₹10 lakhs/year<br/>Industry insights: ₹50 lakhs/year<br/>Research partnerships]
        PARTNERSHIPS[Strategic Partnerships<br/>Technology licensing: ₹2 crores<br/>Data co-creation: ₹5 crores<br/>Joint ventures]
    end

    subgraph "🎯 COST STRUCTURE"
        TECH_INFRA[Technology Infrastructure<br/>Cloud hosting: ₹20 lakhs/month<br/>API credits: ₹10 lakhs/month<br/>ML training: ₹5 lakhs/month]
        TEAM_OPERATIONS[Team & Operations<br/>25-person team: ₹1 crore/year<br/>Field operations: ₹50 lakhs/year<br/>Tech development: ₹2 crores/year]
        PARTNER_COMMISSION[Partner Commissions<br/>Input company margins: 15%<br/>Cooperative revenue share: 20%<br/>Government contracts: 25%]
        MARKETING_COMMS[Marketing & Communications<br/>Digital marketing: ₹50 lakhs/year<br/>Field demonstrations: ₹1 crore/year<br/>Brand awareness: ₹2 crores/year]
    end

    subgraph "📊 REVENUE PROJECTIONS"
        Q1_2025[Year 1: ₹10 crores<br/>2M farmers<br/>Break-even]
        Q2_2026[Year 2: ₹40 crores<br/>8M farmers<br/>20% margin]
        Q3_2027[Year 3: ₹120 crores<br/>15M farmers<br/>30% margin]
        Q4_2028[Year 4: ₹300 crores<br/>25M farmers<br/>35% margin]
        Q5_2029[Year 5: ₹750 crores<br/>30M farmers<br/>40% margin]
    end

    subgraph "🏗️ SCALING INFRASTRUCTURE"
        FREEMIUM_USERS[Free Tier<br/>60% adoption<br/>Retention: 85%<br/>Conversion: 3-5%]
        PREMIUM_USERS[Premium Tier<br/>₹500/month<br/>Freemium conversion<br/>40% of active users]
        ENTERPRISE_USERS[Enterprise Tier<br/>₹5000/bulk<br/>Coop/govt contracts<br/>5-10% of premium]
        GOVT_CONTRACTS[Government Contracts<br/>₹5-50 crores/year<br/>Digital India partnership<br/>R&D partnerships]
    end

    subgraph "🎯 SCALING PARTNERSHIPS"
        IFFCO[IFFCO Partnership<br/>25M cooperative members<br/>₹50 crore potential<br/>Input supply integration]
        NAFED[NAFED Alliance<br/>Market intelligence platform<br/>₹20 crore revenue<br/>MSP optimization]
        BAYER[Bayer CropScience<br/>Digital farming platform<br/>₹10 crore licensing<br/>Responsible agriculture]
        JIO[Jio Platforms<br/>800M user base<br/>₹30 crore marketplace<br/>Farmer network effects]
        AIRtel[AirTel Digital<br/>Payments infrastructure<br/>₹5 crore partnership<br/>Digital lending]
    end

    subgraph "📈 OUTCOME METRICS"
        FARMER_IMPACT[Farm Income Increase<br/>₹25,000-50,000/farmer<br/>15-25% yield improvement<br/>40% risk reduction]
        ECONOMIC_MULTIPLIER[Economic Impact<br/>₹75,000 crores farmer earnings<br/>₹2.25 lakh crores value chain<br/>₹25,000 crores new jobs]
        MARKET_PENETRATION[Market Leadership<br/>30M farmers (20% India households)<br/>₹750 crores annual revenue<br/>Category leader positioning]
    end

    %% Relationships
    SMALLHOLDER --> FREEMIUM_VAL
    PROGRESSIVE --> PREMIUM_VAL
    COOPERATIVE --> ENTERPRISE_VAL
    AGRI_CORP --> PARTNER_VAL

    FREEMIUM_VAL --> SUBSCRIPTION
    PREMIUM_VAL --> SUBSCRIPTION
    ENTERPRISE_VAL --> SUBSCRIPTION
    PARTNER_VAL --> TRANSACTIONS

    TECH_INFRA --> TECH_INFRA
    TEAM_OPERATIONS --> TEAM_OPERATIONS
    PARTNER_COMMISSION --> PARTNER_COMMISSION
    MARKETING_COMMS --> MARKETING_COMMS

    Q1_2025 --> Q2_2026
    Q2_2026 --> Q3_2027
    Q3_2027 --> Q4_2028
    Q4_2028 --> Q5_2029

    FREEMIUM_USERS --> PREMIUM_USERS
    PREMIUM_USERS --> ENTERPRISE_USERS
    ENTERPRISE_USERS --> GOVT_CONTRACTS

    IFFCO --> IFFCO
    NAFED --> NAFED
    BAYER --> BAYER
    JIO --> JIO
    AIRtel --> AIRtel

    FARMER_IMPACT --> ECONOMIC_MULTIPLIER
    ECONOMIC_MULTIPLIER --> MARKET_PENETRATION

    classDef customers fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef valueProps fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef revenue fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef costs fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef projections fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef scaling fill:#fff8e1,stroke:#ff9800,stroke-width:2px
    classDef partnerships fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef outcomes fill:#f1f8e9,stroke:#558b2f,stroke-width:2px

    class SMALLHOLDER,PROGRESSIVE,COOPERATIVE,AGRI_CORP customers
    class FREEMIUM_VAL,PREMIUM_VAL,ENTERPRISE_VAL,PARTNER_VAL valueProps
    class SUBSCRIPTION,TRANSACTIONS,DATA_LICENSING,PARTNERSHIPS revenue
    class TECH_INFRA,TEAM_OPERATIONS,PARTNER_COMMISSION,MARKETING_COMMS costs
    class Q1_2025,Q2_2026,Q3_2027,Q4_2028,Q5_2029 projections
    class FREEMIUM_USERS,PREMIUM_USERS,ENTERPRISE_USERS,GOVT_CONTRACTS scaling
    class IFFCO,NAFED,BAYER,JIO,AIRtel partnerships
    class FARMER_IMPACT,ECONOMIC_MULTIPLIER,MARKET_PENETRATION outcomes
