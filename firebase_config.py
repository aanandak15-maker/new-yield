#!/usr/bin/env python3
"""
Firebase Configuration for Agricultural Intelligence Platform
Provides Firebase Authentication, Firestore Database, and Cloud Functions integration
"""

import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from firebase_admin import functions as firebase_functions
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)

class FirebaseManager:
    """Firebase integration manager for the agricultural platform"""

    def __init__(self):
        # Firebase configuration
        self.project_id = os.getenv('FIREBASE_PROJECT_ID', 'agricultural-intelligence-platform')
        self.api_key = os.getenv('FIREBASE_API_KEY', 'AIzaSyDummyApiKeyForDevelopment')
        self.auth_domain = f"{self.project_id}.firebaseapp.com"
        self.firebase_available = False  # Initialize as False

        # Initialize Firebase Admin SDK
        try:
            # Check if already initialized
            firebase_admin.get_app()
            self.firebase_available = True
        except ValueError:
            # Initialize with credentials (use service account for server-side)
            # For MVP, we'll use a mock initialization
            try:
                # Try to use service account if available
                cred_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
                if cred_path and os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred, {
                        'projectId': self.project_id,
                        'storageBucket': f"{self.project_id}.appspot.com"
                    })
                    logger.info("✅ Firebase initialized with service account")
                else:
                    # Initialize without credentials (limited functionality)
                    firebase_admin.initialize_app(options={
                        'projectId': self.project_id
                    })
                    logger.warning("⚠️ Firebase initialized without credentials (limited functionality)")
            except Exception as e:
                logger.error(f"❌ Firebase initialization failed: {e}")
                self.firebase_available = False
                return

        # Initialize clients (only if credentials are available)
        if self.firebase_available:
            try:
                self.db = firestore.client()
                self.bucket = storage.bucket(f"{self.project_id}.appspot.com")
                logger.info("✅ Firebase clients initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Firebase clients initialization failed (limited functionality): {e}")
                self.db = None
                self.bucket = None
                self.firebase_available = False

        logger.info("✅ Firebase Manager initialized successfully")

    # Authentication Methods
    def create_farmer_user(self, phone_number: str, display_name: str = None, email: str = None) -> dict:
        """Create a new farmer user account"""
        if not self.firebase_available:
            return self._create_mock_user(phone_number, display_name, email)

        try:
            user = auth.create_user(
                phone_number=phone_number,
                display_name=display_name or f"Farmer {phone_number[-4:]}",
                email=email,
                email_verified=False
            )

            logger.info(f"✅ Created Firebase user: {user.uid} for {phone_number}")
            return {
                'success': True,
                'uid': user.uid,
                'phone_number': phone_number,
                'created_at': datetime.now().isoformat()
            }

        except auth.AuthError as e:
            logger.error(f"❌ Failed to create Firebase user: {e}")
            return {
                'success': False,
                'error': str(e),
                'phone_number': phone_number
            }

    def verify_firebase_token(self, token: str) -> dict:
        """Verify Firebase authentication token"""
        if not self.firebase_available:
            return {'valid': True, 'uid': 'mock_uid', 'claims': {}}

        try:
            decoded_token = auth.verify_id_token(token)
            return {
                'valid': True,
                'uid': decoded_token['uid'],
                'phone_number': decoded_token.get('phone_number'),
                'claims': decoded_token
            }
        except Exception as e:
            logger.warning(f"❌ Invalid Firebase token: {e}")
            return {
                'valid': False,
                'error': str(e)
            }

    # Database Methods
    def save_consultation(self, consultation_data: dict) -> bool:
        """Save agricultural consultation to Firestore"""
        if not self.firebase_available:
            logger.info("📝 Mock saved consultation to Firestore")
            return True

        try:
            consultation_id = consultation_data.get('consultation_id', f"consult_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            doc_ref = self.db.collection('consultations').document(consultation_id)
            doc_ref.set({
                **consultation_data,
                'firestore_saved_at': datetime.now().isoformat(),
                'source': 'agricultural_platform'
            })

            logger.info(f"💾 Saved consultation {consultation_id} to Firestore")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save consultation to Firestore: {e}")
            return False

    def save_farmer_profile(self, farmer_data: dict) -> bool:
        """Save farmer profile to Firestore"""
        if not self.firebase_available:
            logger.info("👤 Mock saved farmer profile to Firestore")
            return True

        try:
            uid = farmer_data.get('uid', farmer_data.get('phone_number', 'unknown'))

            doc_ref = self.db.collection('farmers').document(uid)
            doc_ref.set({
                **farmer_data,
                'updated_at': datetime.now().isoformat(),
                'platform_version': '2.0.0'
            })

            logger.info(f"👤 Saved farmer profile {uid} to Firestore")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save farmer profile: {e}")
            return False

    def get_farmer_consultations(self, uid: str, limit: int = 50) -> list:
        """Get farmer consultation history from Firestore"""
        if not self.firebase_available:
            return []

        try:
            docs = self.db.collection('consultations') \
                         .where('farmer_id', '==', uid) \
                         .order_by('created_at', direction=firestore.Query.DESCENDING) \
                         .limit(limit) \
                         .stream()

            consultations = [doc.to_dict() for doc in docs]
            logger.info(f"📚 Retrieved {len(consultations)} consultations for farmer {uid}")
            return consultations

        except Exception as e:
            logger.error(f"❌ Failed to get farmer consultations: {e}")
            return []

    def save_feedback(self, feedback_data: dict) -> bool:
        """Save farmer feedback for continuous learning"""
        if not self.firebase_available:
            logger.info("📝 Mock saved feedback to Firestore")
            return True

        try:
            consultation_id = feedback_data.get('consultation_id')
            feedback_id = f"fb_{consultation_id}_{datetime.now().strftime('%H%M%S')}"

            # Save feedback
            self.db.collection('feedback').document(feedback_id).set({
                **feedback_data,
                'processed_for_learning': False,
                'created_at': datetime.now().isoformat()
            })

            # Update consultation with feedback
            consultation_ref = self.db.collection('consultations').document(consultation_id)
            consultation_ref.update({
                'feedback_received': True,
                'feedback_summary': feedback_data.get('feedback_text', '')[:200]
            })

            logger.info(f"📝 Saved feedback {feedback_id} for consultation {consultation_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save feedback: {e}")
            return False

    # Analytics Methods
    def log_platform_analytics(self, event_type: str, event_data: dict):
        """Log platform analytics events"""
        if not self.firebase_available:
            return

        try:
            event_id = f"evt_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.db.collection('analytics').document(event_id).set({
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': datetime.now().isoformat(),
                'platform': 'agricultural_intelligence'
            })

            logger.info(f"📊 Logged analytics event: {event_type}")

        except Exception as e:
            logger.error(f"❌ Failed to log analytics: {e}")

    # Mock methods for development without Firebase service account
    def _create_mock_user(self, phone_number: str, display_name: str = None, email: str = None) -> dict:
        """Create mock user for development (without Firebase credentials)"""
        import hashlib
        uid = hashlib.md5(phone_number.encode()).hexdigest()[:16]

        logger.info(f"🎭 Mock created user {uid} for {phone_number}")

        return {
            'success': True,
            'uid': uid,
            'phone_number': phone_number,
            'mock_mode': True,
            'created_at': datetime.now().isoformat()
        }

    def get_system_metrics(self) -> dict:
        """Get system usage metrics"""
        return {
            'firebase_available': self.firebase_available,
            'project_id': self.project_id,
            'firestore_available': self.firebase_available,
            'authentication_available': self.firebase_available,
            'storage_available': self.firebase_available,
            'last_checked': datetime.now().isoformat()
        }

# Global Firebase manager instance
firebase_manager = FirebaseManager()

if __name__ == "__main__":
    print("🔥 FIREBASE CONFIGURATION FOR AGRICULTURAL INTELLIGENCE PLATFORM")
    print("=" * 70)

    # Test Firebase integration
    metrics = firebase_manager.get_system_metrics()
    print(f"📊 Firebase Status: {'✅ Operational' if metrics['firebase_available'] else '⚠️ Limited Mode'}")
    print(f"🏗️ Project ID: {metrics['project_id']}")
    print(f"🗄️ Firestore: {'✅ Available' if metrics['firestore_available'] else '❌ Not Available'}")
    print(f"🔐 Authentication: {'✅ Available' if metrics['authentication_available'] else '❌ Not Available'}")

    if metrics['firebase_available']:
        # Test farmer user creation
        test_result = firebase_manager.create_farmer_user("+919876543210", "Test Farmer")
        print(f"👤 Test User Creation: {'✅ Success' if test_result.get('success') else '❌ Failed'}")
        print(f"🆔 User ID: {test_result.get('uid', 'N/A')}")

        print("\n🎉 Firebase integration ready for agricultural platform!")
        print("🚀 Ready for farmer authentication, consultation storage, and analytics!")
    else:
        print("\n⚠️ Running in development mode with limited Firebase functionality")
        print("📝 To enable full features, add Firebase service account credentials")
        print("🔧 Set FIREBASE_SERVICE_ACCOUNT_KEY environment variable")
