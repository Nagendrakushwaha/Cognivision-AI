import React, { useEffect, useState } from 'react';
import {
  Database,
  Cpu,
  Award,
  Layers,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Activity,
  CheckCircle2,
  Clock
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid
} from 'recharts';
import MetricCard from '../components/MetricCard';
import { api } from '../services/api';

export default function Dashboard({ setActiveTab }) {
  const [summary, setSummary] = useState(null);
  const [perf, setPerf] = useState(null);
  const [trainStatus, setTrainStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [sumRes, perfRes, statusRes] = await Promise.all([
          api.getDatasetSummary(),
          api.getModelPerformance(),
          api.getTrainingStatus()
        ]);
        setSummary(sumRes);
        setPerf(perfRes);
        setTrainStatus(statusRes);
      } catch (err) {
        console.error('Error loading dashboard data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading Cognivision AI Dashboard...
      </div>
    );
  }

  // Prepare chart data for class distribution
  const chartData = summary ? summary.classes.map(cls => ({
    name: cls.length > 15 ? cls.substring(0, 14) + '...' : cls,
    fullName: cls,
    train: summary.class_distribution_train[cls] || 0,
    test: summary.class_distribution_test[cls] || 0
  })) : [];

  const isModelTrained = perf && perf.is_trained;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      {/* Hero Welcome Banner */}
      <div className="glass-panel" style={{
        padding: '28px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%)',
        borderLeft: '4px solid #8B5CF6'
      }}>
        <div>
          <div className="badge badge-purple" style={{ marginBottom: '8px' }}>
            Production AI Pipeline • SROIE Benchmark
          </div>
          <h1 style={{ fontSize: '1.65rem', fontWeight: 800, margin: '4px 0 8px 0' }}>
            Document Vision & Intelligence Platform
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '640px', margin: 0 }}>
            End-to-end multimodal deep learning system for receipt document categorization, OCR token analysis,
            explainable gradient heatmaps (Grad-CAM), and low-latency inference on local CPU.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-primary" onClick={() => setActiveTab('predictions')}>
            Test Prediction <ArrowRight size={16} />
          </button>
          <button className="btn-secondary" onClick={() => setActiveTab('explainability')}>
            <Sparkles size={16} color="#A78BFA" /> View Grad-CAM
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '20px'
      }}>
        <MetricCard
          title="Total Receipts"
          value={summary ? summary.total_records : '987'}
          subtext={`${summary ? summary.train_records : 626} Train / ${summary ? summary.test_records : 361} Test`}
          icon={Database}
          color="#8B5CF6"
        />
        <MetricCard
          title="Target Classes"
          value={summary ? summary.num_classes : '6'}
          subtext="Balanced Merchant Categories"
          icon={Layers}
          color="#06B6D4"
        />
        <MetricCard
          title="Model Status"
          value={isModelTrained ? 'Active' : 'Untrained'}
          subtext={isModelTrained ? `${perf.model_name} • Checkpoint Loaded` : 'Ready to train'}
          icon={Cpu}
          color={isModelTrained ? '#10B981' : '#F59E0B'}
        />
        <MetricCard
          title="Test Accuracy"
          value={isModelTrained ? `${(perf.accuracy * 100).toFixed(1)}%` : 'N/A'}
          subtext={isModelTrained ? `Macro F1: ${(perf.macro_f1 * 100).toFixed(1)}%` : 'Evaluate on test set'}
          icon={Award}
          color="#F43F5E"
        />
      </div>

      {/* Main Visualizations Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        {/* Class Distribution Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Dataset Distribution by Category</h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                Comparing sample frequencies across Train (626) and Test (361) sets
              </p>
            </div>
            <button className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.75rem' }} onClick={() => setActiveTab('data-analysis')}>
              Deep Analysis
            </button>
          </div>

          <div style={{ width: '100%', height: '280px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="var(--text-dim)" fontSize={11} angle={-15} textAnchor="end" />
                <YAxis stroke="var(--text-dim)" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    fontSize: '0.78rem'
                  }}
                  formatter={(value, name) => [value, name === 'train' ? 'Train Samples' : 'Test Samples']}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="train" name="Train Set" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="test" name="Test Set" fill="#06B6D4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Performance Snapshot */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Model Performance</h3>
              <span className={`badge ${isModelTrained ? 'badge-emerald' : 'badge-amber'}`}>
                {isModelTrained ? 'Validated' : 'Pending'}
              </span>
            </div>

            {isModelTrained ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Accuracy:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#10B981' }}>{(perf.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Macro F1:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#A78BFA' }}>{(perf.macro_f1 * 100).toFixed(2)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Weighted F1:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#67E8F9' }}>{(perf.weighted_f1 * 100).toFixed(2)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>ROC-AUC (OVR):</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#FCD34D' }}>{perf.roc_auc ? perf.roc_auc.toFixed(4) : 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Test Samples:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#FFFFFF' }}>{perf.evaluated_samples}</span>
                </div>
              </div>
            ) : (
              <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.85rem' }}>
                Model not trained yet. Visit the Model Training page to initiate local training.
              </div>
            )}
          </div>

          <button
            className="btn-primary"
            style={{ width: '100%', marginTop: '20px' }}
            onClick={() => setActiveTab('model-performance')}
          >
            Open Evaluation Dashboard <ArrowRight size={15} />
          </button>
        </div>
      </div>

      {/* Quick Action Navigation Grid */}
      <div>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '14px', color: '#F3F4F6' }}>
          Explore System Capabilities
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '16px'
        }}>
          <div
            className="glass-panel"
            onClick={() => setActiveTab('dataset-explorer')}
            style={{ padding: '20px', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Database size={20} color="#8B5CF6" />
              <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600 }}>Dataset Explorer</h4>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Browse all 987 receipt records, dimensions, extracted entities, and OCR words interactively.
            </p>
          </div>

          <div
            className="glass-panel"
            onClick={() => setActiveTab('model-training')}
            style={{ padding: '20px', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Cpu size={20} color="#06B6D4" />
              <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600 }}>Model Training</h4>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Fine-tune lightweight MobileNetV3 or train CogniNet CNN on your laptop CPU with live tracking.
            </p>
          </div>

          <div
            className="glass-panel"
            onClick={() => setActiveTab('explainability')}
            style={{ padding: '20px', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Sparkles size={20} color="#10B981" />
              <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600 }}>Explainable AI</h4>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Inspect Grad-CAM gradient activation heatmaps overlaid on receipt images with regional breakdown.
            </p>
          </div>

          <div
            className="glass-panel"
            onClick={() => setActiveTab('multimodal-kie')}
            style={{ padding: '20px', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <Layers size={20} color="#F59E0B" />
              <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600 }}>Multimodal KIE</h4>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Inspect 2D bounding boxes and extracted entities (Company, Date, Address, Total) from SROIE OCR.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
