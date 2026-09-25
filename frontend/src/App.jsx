import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import DatasetExplorer from './pages/DatasetExplorer';
import DataAnalysis from './pages/DataAnalysis';
import ModelTraining from './pages/ModelTraining';
import ModelPerformance from './pages/ModelPerformance';
import Prediction from './pages/Prediction';
import BatchPrediction from './pages/BatchPrediction';
import Explainability from './pages/Explainability';
import MultimodalKIE from './pages/MultimodalKIE';
import AboutProject from './pages/AboutProject';
import { api } from './services/api';

const TAB_TITLES = {
  'dashboard': 'Executive Dashboard',
  'dataset-explorer': 'SROIE Dataset Explorer',
  'data-analysis': 'Data Analysis & Visualizations',
  'model-training': 'Model Training Pipeline',
  'model-performance': 'Model Evaluation & Benchmarks',
  'predictions': 'Real-Time Document Inference',
  'batch-prediction': 'Batch Prediction Processing',
  'explainability': 'Explainable AI (Grad-CAM)',
  'multimodal-kie': 'Multimodal Key Information Extraction',
  'about': 'Technical Specifications & Architecture'
};

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [health, setHealth] = useState({ status: 'checking', model_trained: false, active_model: null });

  const checkHealth = async () => {
    try {
      const h = await api.getHealth();
      setHealth(h);
    } catch (err) {
      console.warn('Backend health check warning:', err);
    }
  };

  useEffect(() => {
    checkHealth();
  }, [activeTab]);

  function renderPage() {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard setActiveTab={setActiveTab} />;
      case 'dataset-explorer':
        return <DatasetExplorer onSelectSampleForPrediction={() => setActiveTab('predictions')} />;
      case 'data-analysis':
        return <DataAnalysis />;
      case 'model-training':
        return <ModelTraining onTrainingComplete={() => { checkHealth(); setActiveTab('model-performance'); }} />;
      case 'model-performance':
        return <ModelPerformance setActiveTab={setActiveTab} />;
      case 'predictions':
        return <Prediction />;
      case 'batch-prediction':
        return <BatchPrediction />;
      case 'explainability':
        return <Explainability />;
      case 'multimodal-kie':
        return <MultimodalKIE />;
      case 'about':
        return <AboutProject />;
      default:
        return <Dashboard setActiveTab={setActiveTab} />;
    }
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-primary)' }}>
      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Navbar
          activeTitle={TAB_TITLES[activeTab] || 'Cognivision AI'}
          isBackendHealthy={health.status === 'healthy'}
          modelTrained={health.model_trained}
          activeModelName={health.active_model?.architecture}
        />
        <main style={{ flex: 1, overflowY: 'auto' }}>
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
