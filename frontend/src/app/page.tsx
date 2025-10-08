"use client";

import { useState, useEffect } from 'react';
import {
  MapPin,
  Plus,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Satellite,
  Eye,
  Sparkles,
  Leaf,
  BarChart3,
  Settings,
  Bell,
  User,
  Home,
  RefreshCw,
  Globe,
  Zap
} from 'lucide-react';

interface PredictionRequest {
  latitude: number;
  longitude: number;
  crop?: string;
  season?: string;
}

interface PredictionResponse {
  crop_type: string;
  predicted_yield_quintal_ha: number;
  confidence_level: string;
  location_context: string;
  regional_crop_suitability: Record<string, number>;
  alternative_crops: Array<{
    crop: string;
    suitability_score: number;
  }>;
  crop_rotation_suggestions: string[];
  timestamp: string;
}

interface LocationData {
  lat: number;
  lng: number;
  district: string;
  state: string;
  lastPrediction?: PredictionResponse;
}

function App() {
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAddLocation, setShowAddLocation] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [newLocationForm, setNewLocationForm] = useState({
    latitude: '',
    longitude: '',
    crop: '',
    season: 'kharif'
  });

  const locations: LocationData[] = [
    {
      lat: 30.7333,
      lng: 76.8142,
      district: 'Hoshiarpur',
      state: 'Punjab',
      lastPrediction: predictions.find(p => p.timestamp.includes('30.7333'))
    },
    {
      lat: 28.6139,
      lng: 77.2090,
      district: 'Delhi',
      state: 'NCT',
      lastPrediction: predictions.find(p => p.timestamp.includes('28.6139'))
    },
    {
      lat: 22.2587,
      lng: 71.1924,
      district: 'Ahmedabad',
      state: 'Gujarat',
      lastPrediction: predictions.find(p => p.timestamp.includes('22.2587'))
    },
    {
      lat: 27.0238,
      lng: 74.2179,
      district: 'Ajmer',
      state: 'Rajasthan',
      lastPrediction: predictions.find(p => p.timestamp.includes('27.0238'))
    }
  ];

  const checkBackendHealth = async () => {
    try {
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
      const response = await fetch(`${apiBaseUrl}/health`);
      if (response.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    } catch (error) {
      setBackendStatus('offline');
    }
  };

  const getYieldPrediction = async (req: PredictionRequest): Promise<PredictionResponse> => {
    try {
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
      const response = await fetch(`${apiBaseUrl}/unified/predict/yield`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(req),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  };

  const handleLocationPrediction = async (locationData: LocationData) => {
    setLoading(true);

    try {
      const request: PredictionRequest = {
        latitude: locationData.lat,
        longitude: locationData.lng,
        season: 'kharif'
      };

      const prediction = await getYieldPrediction(request);

      const predictionWithLocation = {
        ...prediction,
        location_lat: locationData.lat,
        location_lng: locationData.lng,
        location_district: locationData.district
      };

      setPredictions(prev => [predictionWithLocation, ...prev].slice(0, 10));
      setSelectedLocation({ ...locationData, lastPrediction: prediction });
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Failed to get yield prediction. Make sure FastAPI backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleFormPrediction = async () => {
    if (!newLocationForm.latitude || !newLocationForm.longitude) {
      alert('Please enter both latitude and longitude');
      return;
    }

    setLoading(true);

    try {
      const request: PredictionRequest = {
        latitude: parseFloat(newLocationForm.latitude),
        longitude: parseFloat(newLocationForm.longitude),
        crop: newLocationForm.crop || undefined,
        season: newLocationForm.season as 'kharif' | 'rabi' | 'zaid'
      };

      const prediction = await getYieldPrediction(request);

      const predictionWithLocation = {
        ...prediction,
        location_lat: parseFloat(newLocationForm.latitude),
        location_lng: parseFloat(newLocationForm.longitude)
      };

      setPredictions(prev => [predictionWithLocation, ...prev].slice(0, 10));
      setShowAddLocation(false);
      setNewLocationForm({ latitude: '', longitude: '', crop: '', season: 'kharif' });
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Failed to get yield prediction. Check backend connectivity.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000); // Check every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (prediction: PredictionResponse) => {
    if (prediction.confidence_level === 'High') {
      return <CheckCircle className="w-4 h-4 text-green-600" />;
    } else if (prediction.confidence_level === 'Medium') {
      return <Clock className="w-4 h-4 text-yellow-600" />;
    } else {
      return <AlertTriangle className="w-4 h-4 text-red-600" />;
    }
  };

  const getStatusBadge = (prediction: PredictionResponse) => {
    const colors = {
      High: 'bg-green-100 text-green-800 border-green-200',
      Medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      Low: 'bg-red-100 text-red-800 border-red-200'
    };

    return colors[prediction.confidence_level as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const getBackendStatusColor = () => {
    switch (backendStatus) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-yellow-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-lg border-b border-green-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-green-600 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
                <Leaf className="text-white w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">India Agri Intelligence</h1>
                <p className="text-sm text-green-600 font-medium">Real ML Predictions & Analytics</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Predictions Made</p>
                <p className="text-lg font-semibold text-gray-900">{predictions.length}</p>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full animate-pulse ${getBackendStatusColor()}`}></div>
                <span className="text-sm text-gray-600 capitalize">Backend: {backendStatus}</span>
              </div>
              <button
                onClick={checkBackendHealth}
                className="p-2 bg-green-100 hover:bg-green-200 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4 text-green-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Main Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Location Cards */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Agricultural Locations</h2>
              <button
                onClick={() => setShowAddLocation(true)}
                className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Add Location</span>
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {locations.map((location, index) => (
                <div key={index} className="bg-white rounded-xl shadow-lg p-6 border border-green-100 hover:shadow-xl transition-shadow">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-5 h-5 text-green-600" />
                      <div>
                        <h3 className="font-semibold text-gray-900">{location.district}</h3>
                        <p className="text-sm text-gray-600">{location.state}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          {location.lat.toFixed(4)}°N, {location.lng.toFixed(4)}°E
                        </p>
                      </div>
                    </div>
                    {location.lastPrediction && getStatusIcon(location.lastPrediction)}
                  </div>

                  <div className="mt-4 flex items-center justify-between">
                    {location.lastPrediction ? (
                      <div>
                        <p className="text-lg font-bold text-green-600">
                          {location.lastPrediction.predicted_yield_quintal_ha} q/ha
                        </p>
                        <p className="text-sm text-gray-600 capitalize">
                          {location.lastPrediction.crop_type}
                        </p>
                      </div>
                    ) : (
                      <p className="text-gray-500 text-sm">No prediction yet</p>
                    )}

                    <button
                      onClick={() => handleLocationPrediction(location)}
                      disabled={loading || backendStatus !== 'online'}
                      className="bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 disabled:opacity-50 transition-colors"
                    >
                      {loading ? 'Predicting...' : 'Get Prediction'}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Recent Predictions */}
            {predictions.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">Recent Predictions</h3>
                  <p className="text-sm text-gray-600">Latest ML model outputs</p>
                </div>
                <div className="divide-y divide-gray-100">
                  {predictions.slice(0, 5).map((pred, index) => (
                    <div key={index} className="px-6 py-4 hover:bg-gray-50">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(pred)}
                          <div>
                            <p className="font-medium text-gray-900 capitalize">{pred.crop_type}</p>
                            <p className="text-sm text-gray-600">
                              {(pred as any).location_lat?.toFixed(2) || 'Location'}°N, {(pred as any).location_lng?.toFixed(2) || 'Location'}°E
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold text-green-600">{pred.predicted_yield_quintal_ha} q/ha</p>
                          <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full border ${getStatusBadge(pred)}`}>
                            {pred.confidence_level} confidence
                          </span>
                        </div>
                      </div>
                      {pred.alternative_crops?.length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs text-gray-500">Alternatives: {pred.alternative_crops.slice(0, 3).map(a => a.crop).join(', ')}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Backend API</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${getBackendStatusColor()}`}></div>
                    <span className="text-sm capitalize">{backendStatus}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">ML Models</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm">Loaded</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Predictions</span>
                  <span className="text-sm font-medium">{predictions.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Frontend</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm">Running</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Supported Crops</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Rice</span>
                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Available</div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Wheat</span>
                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Available</div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Cotton</span>
                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Available</div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Maize</span>
                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Available</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={() => setShowAddLocation(true)}
                  disabled={backendStatus !== 'online'}
                  className="w-full flex items-center justify-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span>New Prediction</span>
                </button>
                <button
                  onClick={checkBackendHealth}
                  className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Check Backend</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Add Location Modal */}
        {showAddLocation && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl max-w-md w-full">
              <div className="p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">New Location Prediction</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Latitude</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={newLocationForm.latitude}
                      onChange={(e) => setNewLocationForm({...newLocationForm, latitude: e.target.value})}
                      placeholder="28.6139"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Longitude</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={newLocationForm.longitude}
                      onChange={(e) => setNewLocationForm({...newLocationForm, longitude: e.target.value})}
                      placeholder="77.2090"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Preferred Crop (Optional)</label>
                    <select
                      value={newLocationForm.crop}
                      onChange={(e) => setNewLocationForm({...newLocationForm, crop: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    >
                      <option value="">Auto-detect crop</option>
                      <option value="rice">Rice</option>
                      <option value="wheat">Wheat</option>
                      <option value="cotton">Cotton</option>
                      <option value="maize">Maize</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Season</label>
                    <select
                      value={newLocationForm.season}
                      onChange={(e) => setNewLocationForm({...newLocationForm, season: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    >
                      <option value="kharif">Kharif</option>
                      <option value="rabi">Rabi</option>
                      <option value="zaid">Zaid</option>
                    </select>
                  </div>
                </div>

                <div className="flex space-x-3 mt-6">
                  <button
                    onClick={() => setShowAddLocation(false)}
                    className="flex-1 bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleFormPrediction}
                    disabled={loading || backendStatus !== 'online'}
                    className="flex-1 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50"
                  >
                    {loading ? 'Getting ML Prediction...' : 'Get ML Prediction'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <footer className="bg-white/80 backdrop-blur-lg border-t border-green-100 mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center space-x-4">
              <span className="flex items-center space-x-2">
                <Satellite className="w-4 h-4" />
                <span>Real ML Models</span>
              </span>
              <span className="flex items-center space-x-2">
                <Leaf className="w-4 h-4" />
                <span>Geospatial Intelligence</span>
              </span>
              <span className="flex items-center space-x-2">
                <Sparkles className="w-4 h-4" />
                <span>Gemini AI Assistant</span>
              </span>
            </div>
            <span className="text-xs text-gray-500">
              India Agricultural Intelligence Platform | Multi-Crop AI Analytics
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
