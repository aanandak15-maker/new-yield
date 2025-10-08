#!/usr/bin/env python3
"""
Railway Configuration for Agricultural Intelligence Platform
Handles Railway PostgreSQL database connections and deployments
"""

import os
import logging
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class RailwayDatabaseManager:
    """Railway PostgreSQL database manager for the agricultural platform"""

    def __init__(self):
        # Railway database configuration
        self.database_url = os.getenv('DATABASE_URL', os.getenv('RAILWAY_DATABASE_URL'))

        if self.database_url:
            logger.info("✅ Railway database URL detected")
            self.db_available = True
            self.connection_pool = []
            self.initialize_connection_pool()
        else:
            logger.warning("⚠️ Railway database URL not found - using SQLite fallback")
            self.db_available = False
            self.sqlite_path = os.getenv('SQLITE_DATABASE_PATH', 'agriculture_platform.db')
            self.initialize_sqlite_fallback()

        logger.info(f"🏗️ Railway Database Status: {'✅ Connected' if self.get_health_status() else '❌ Not Connected'}")

    def initialize_connection_pool(self, pool_size: int = 5):
        """Initialize PostgreSQL connection pool"""
        if not self.database_url:
            return

        try:
            parsed_url = urlparse(self.database_url)

            # Extract connection parameters from URL
            self.db_config = {
                'host': parsed_url.hostname,
                'port': parsed_url.port,
                'database': parsed_url.path.lstrip('/'),
                'user': parsed_url.username,
                'password': parsed_url.password
            }

            # Create connection pool
            for i in range(pool_size):
                conn = psycopg2.connect(**self.db_config)
                conn.autocommit = True  # Enable autocommit for better performance
                self.connection_pool.append(conn)

            logger.info(f"✅ Railway connection pool initialized with {len(self.connection_pool)} connections")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Railway connection pool: {e}")
            self.db_available = False

    def initialize_sqlite_fallback(self):
        """Initialize SQLite fallback database for development"""
        import sqlite3

        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            self.sqlite_conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            self.sqlite_conn.execute("PRAGMA foreign_keys=ON")   # Enable foreign keys
            self.create_sqlite_tables()
            logger.info(f"📱 SQLite fallback database initialized at {self.sqlite_path}")
        except Exception as e:
            logger.error(f"❌ SQLite fallback initialization failed: {e}")

    def create_sqlite_tables(self):
        """Create SQLite tables for fallback database"""
        if not hasattr(self, 'sqlite_conn'):
            return

        try:
            # Create tables matching PostgreSQL schema
            tables = [
                """
                CREATE TABLE IF NOT EXISTS farmer_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    firebase_uid TEXT UNIQUE NOT NULL,
                    phone_number TEXT NOT NULL,
                    display_name TEXT,
                    location TEXT,
                    farm_size REAL,
                    district TEXT,
                    state TEXT,
                    pincode TEXT,
                    preferred_crops TEXT,
                    subscription_tier TEXT DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS consultations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consultation_id TEXT UNIQUE NOT NULL,
                    farmer_id TEXT NOT NULL,
                    firebase_uid TEXT,
                    query_text TEXT NOT NULL,
                    crop_type TEXT,
                    problem_description TEXT,
                    solution_recommended TEXT,
                    ethical_tier_applied TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_status TEXT DEFAULT 'completed'
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consultation_id TEXT NOT NULL,
                    farmer_uid TEXT NOT NULL,
                    feedback_text TEXT,
                    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                    helpful_boolean INTEGER DEFAULT 0,
                    processed_for_learning INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(consultation_id) REFERENCES consultations(consultation_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    farmer_uid TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                )
                """
            ]

            for table_sql in tables:
                self.sqlite_conn.execute(table_sql)

            self.sqlite_conn.commit()
            logger.info("📱 SQLite tables created successfully")

        except Exception as e:
            logger.error(f"❌ SQLite table creation failed: {e}")

    def get_connection(self):
        """Get a database connection from pool"""
        if not self.db_available:
            return self.sqlite_conn

        if self.connection_pool:
            return self.connection_pool[0]  # Simple round-robin can be improved
        else:
            # Fallback - create new connection
            try:
                return psycopg2.connect(**self.db_config)
            except Exception as e:
                logger.error(f"❌ Failed to create new Railway connection: {e}")
                return None

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Optional[List]:
        """Execute database query"""
        conn = self.get_connection()
        if not conn:
            logger.error("❌ No database connection available")
            return None

        try:
            if self.db_available:
                # Railway PostgreSQL
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, params or ())
                    if fetch:
                        results = cursor.fetchall()
                        return [dict(row) for row in results] if results else []
                    else:
                        conn.commit()
                        return cursor.rowcount
            else:
                # SQLite fallback
                cursor = conn.execute(query, params or ())
                if fetch:
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    results = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in results] if results else []
                else:
                    conn.commit()
                    return cursor.rowcount

        except Exception as e:
            logger.error(f"❌ Database query failed: {e}")
            return None
        finally:
            if self.db_available and conn not in self.connection_pool:
                conn.close()

    # Farmer Profile Operations
    def create_farmer_profile(self, farmer_data: Dict[str, Any]) -> bool:
        """Create farmer profile in database"""
        query = """
            INSERT INTO farmer_profiles
            (firebase_uid, phone_number, display_name, location, farm_size, district, state, pincode, preferred_crops)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (firebase_uid) DO UPDATE SET
                phone_number = EXCLUDED.phone_number,
                display_name = EXCLUDED.display_name,
                location = EXCLUDED.location,
                farm_size = EXCLUDED.farm_size,
                district = EXCLUDED.district,
                state = EXCLUDED.state,
                pincode = EXCLUDED.pincode,
                preferred_crops = EXCLUDED.preferred_crops,
                updated_at = CURRENT_TIMESTAMP
        """ if self.db_available else """
            INSERT OR REPLACE INTO farmer_profiles
            (firebase_uid, phone_number, display_name, location, farm_size, district, state, pincode, preferred_crops, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            farmer_data['firebase_uid'],
            farmer_data['phone_number'],
            farmer_data.get('display_name'),
            farmer_data.get('location'),
            farmer_data.get('farm_size'),
            farmer_data.get('district'),
            farmer_data.get('state'),
            farmer_data.get('pincode'),
            json.dumps(farmer_data.get('preferred_crops', [])) if farmer_data.get('preferred_crops') else None,
            datetime.now().isoformat() if not self.db_available else None
        )

        result = self.execute_query(query, params, fetch=False)
        if result is not None:
            logger.info(f"👤 Farmer profile created/updated for {farmer_data['firebase_uid']}")
            return True
        return False

    def get_farmer_profile(self, firebase_uid: str) -> Optional[Dict]:
        """Get farmer profile from database"""
        query = "SELECT * FROM farmer_profiles WHERE firebase_uid = %s" if self.db_available else "SELECT * FROM farmer_profiles WHERE firebase_uid = ?"

        results = self.execute_query(query, (firebase_uid,))
        if results and len(results) > 0:
            profile = results[0]
            # Parse JSON fields
            if 'preferred_crops' in profile and profile['preferred_crops']:
                try:
                    profile['preferred_crops'] = json.loads(profile['preferred_crops'])
                except:
                    profile['preferred_crops'] = []
            return profile
        return None

    # Consultation Operations
    def save_consultation(self, consultation_data: Dict[str, Any]) -> bool:
        """Save consultation to database"""
        query = """
            INSERT INTO consultations
            (consultation_id, farmer_id, firebase_uid, query_text, crop_type, problem_description,
             solution_recommended, ethical_tier_applied, confidence_score, response_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (consultation_id) DO UPDATE SET
                solution_recommended = EXCLUDED.solution_recommended,
                ethical_tier_applied = EXCLUDED.ethical_tier_applied,
                confidence_score = EXCLUDED.confidence_score,
                response_status = EXCLUDED.response_status
        """ if self.db_available else """
            INSERT OR REPLACE INTO consultations
            (consultation_id, farmer_id, firebase_uid, query_text, crop_type, problem_description,
             solution_recommended, ethical_tier_applied, confidence_score, response_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            consultation_data['consultation_id'],
            consultation_data['farmer_id'],
            consultation_data.get('firebase_uid'),
            consultation_data['query_text'],
            consultation_data.get('crop_type'),
            consultation_data.get('problem_description'),
            consultation_data.get('solution_recommended'),
            consultation_data.get('ethical_tier_applied'),
            consultation_data.get('confidence_score', 0.0),
            consultation_data.get('response_status', 'completed'),
            datetime.now().isoformat() if not self.db_available else datetime.now()
        )

        result = self.execute_query(query, params, fetch=False)
        if result is not None:
            logger.info(f"💾 Consultation saved: {consultation_data['consultation_id']}")
            return True
        return False

    def get_farmer_consultations(self, firebase_uid: str, limit: int = 50) -> List[Dict]:
        """Get farmer consultation history"""
        query = """
            SELECT * FROM consultations
            WHERE firebase_uid = %s
            ORDER BY created_at DESC
            LIMIT %s
        """ if self.db_available else """
            SELECT * FROM consultations
            WHERE firebase_uid = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        results = self.execute_query(query, (firebase_uid, limit))
        return results or []

    # Feedback Operations
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Save farmer feedback"""
        query = """
            INSERT INTO feedback
            (consultation_id, farmer_uid, feedback_text, rating, helpful_boolean)
            VALUES (%s, %s, %s, %s, %s)
        """ if self.db_available else """
            INSERT INTO feedback
            (consultation_id, farmer_uid, feedback_text, rating, helpful_boolean, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        params = (
            feedback_data['consultation_id'],
            feedback_data['farmer_uid'],
            feedback_data.get('feedback_text'),
            feedback_data.get('rating'),
            feedback_data.get('helpful_boolean', False),
            datetime.now().isoformat() if not self.db_available else datetime.now()
        )

        result = self.execute_query(query, params, fetch=False)
        if result is not None:
            logger.info(f"📝 Feedback saved for consultation {feedback_data['consultation_id']}")
            return True
        return False

    # Analytics Operations
    def log_analytics_event(self, event_data: Dict[str, Any]) -> bool:
        """Log analytics event"""
        query = """
            INSERT INTO analytics_events
            (event_type, event_data, farmer_uid, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s)
        """ if self.db_available else """
            INSERT INTO analytics_events
            (event_type, event_data, farmer_uid, ip_address, user_agent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        params = (
            event_data['event_type'],
            json.dumps(event_data.get('event_data', {})),
            event_data.get('farmer_uid'),
            event_data.get('ip_address'),
            event_data.get('user_agent'),
            datetime.now().isoformat() if not self.db_available else datetime.now()
        )

        result = self.execute_query(query, params, fetch=False)
        if result is not None:
            logger.info(f"📊 Analytics event logged: {event_data['event_type']}")
            return True
        return False

    def get_health_status(self) -> bool:
        """Get database health status"""
        try:
            result = self.execute_query("SELECT 1 as health_check", fetch=True)
            return result is not None and len(result) > 0
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            # Get database size and connection info
            if self.db_available:
                db_type = "Railway PostgreSQL"
                connection_count = len(self.connection_pool)
            else:
                db_type = "SQLite Fallback"
                connection_count = 1

            # Get record counts
            metrics = {}

            for table in ['farmer_profiles', 'consultations', 'feedback', 'analytics_events']:
                try:
                    query = f"SELECT COUNT(*) as count FROM {table}"
                    result = self.execute_query(query, fetch=True)
                    metrics[f"{table}_count"] = result[0]['count'] if result and len(result) > 0 else 0
                except:
                    metrics[f"{table}_count"] = 0

            return {
                'database_type': db_type,
                'available': self.db_available,
                'connection_count': connection_count,
                'health_status': self.get_health_status(),
                'record_counts': metrics,
                'last_checked': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {
                'database_type': 'Unknown',
                'available': False,
                'health_status': False,
                'error': str(e)
            }

    def close_all_connections(self):
        """Close all database connections"""
        if not self.db_available:
            if hasattr(self, 'sqlite_conn'):
                self.sqlite_conn.close()
        else:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool.clear()

# Global Railway database manager instance
railway_db = RailwayDatabaseManager()

if __name__ == "__main__":
    print("🚂 RAILWAY DATABASE CONFIGURATION FOR AGRICULTURAL INTELLIGENCE PLATFORM")
    print("=" * 80)

    # Test Railway database integration
    metrics = railway_db.get_system_metrics()

    print(f"🏗️ Database Type: {metrics['database_type']}")
    print(f"📊 Health Status: {'✅ Operational' if metrics['health_status'] else '❌ Down'}")
    print(f"🔗 Connections: {metrics['connection_count']}")

    if metrics.get('record_counts'):
        print("\n📈 RECORD COUNTS:")
        for table, count in metrics['record_counts'].items():
            print(f"   {table}: {count}")

    print("\n🚀 Railway database integration ready for agricultural platform!")

    if not metrics['available']:
        print("\n📝 Add RAILWAY_DATABASE_URL environment variable for Railway PostgreSQL")
        print("🔧 Using SQLite fallback for development")
    else:
        print("✅ Full Railway database functionality available!")
