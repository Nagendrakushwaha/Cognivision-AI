import React, { useState, useEffect } from 'react';
import {
  Award,
  BarChart2,
  LineChart as LineChartIcon,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  HelpCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
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
import ConfusionMatrix from '../components/ConfusionMatrix';
import { api } from '../services/api';

export default function ModelPerformance({ setActiveTab }) {
  const [perf, setPerf] = useState(null);
  const [trainStatus, setTrainStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    loadPerformance();
  }, []);

  async function loadPerformance() {
    setLoading(true);
    try {
      const [pRes, sRes] = await Promise.all([
        api.getModelPerformance(),
        api.getTrainingStatus()
      ]);
      setPerf(pRes);
      setTrainStatus(sRes);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleReEvaluate() {
    setEvaluating(true);
    try {
      const newPerf = await api.evaluateModel();
      setPerf(newPerf);
    } catch (err) {
      alert('Evaluation failed: ' + err.message);
    } finally {
      setEvaluating(false);
    }
  }

  if (loading) {
    return (
      <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading Model Performance Dashboard...
      </div>
    );
  }

  if (!perf || !perf.is_trained) {
    return (
      <div style={{ padding: '40px', maxWidth: '700px', margin: '60px auto', textAlign: 'center' }}>
        <div className="glass-panel" style={{ padding: '40px' }}>
          <AlertCircle size={44} color="#F59E0B" style={{ marginBottom: '16px' }} />
          <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 10px 0' }}>Model Not Trained Yet</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: 1.6, marginBottom: '24px' }}>
            No model evaluation metrics can be displayed because the computer vision model has not been trained yet.
            Cognivision AI does not fabricate metrics. Please navigate to the Model Training page to train a model on the SROIE receipt dataset.
          </p>
          <button className="btn-primary" onClick={() => setActiveTab('model-training')}>
            Go to Model Training
          </button>
        </div>
      </div>
    );
  }

  // Prepare Accuracy & Loss graphs data from training history
  const historyData = trainStatus && trainStatus.history ? trainStatus.history.map(h => ({
    epoch: `Epoch ${h.epoch}`,
    trainLoss: h.train_loss,
    valLoss: h.val_loss,
    trainAcc: Number((h.train_acc * 100).toFixed(1)),
    valAcc: Number((h.val_acc * 100).toFixed(1))
  })) : [];

  // Prepare Class-wise comparative chart data
  const classWiseData = perf.class_wise_metrics ? perf.class_wise_metrics.map(c => ({
    name: c.class_name.length > 14 ? c.class_name.substring(0, 12) + '...' : c.class_name,
    fullName: c.class_name,
    precision: Number((c.precision * 100).toFixed(1)),
    recall: Number((c.recall * 100).toFixed(1)),
    f1: Number((c.f1_score * 100).toFixed(1)),
    support: c.support
  })) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Model Performance & Evaluation</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
            Authentic evaluation benchmarks calculated across all {perf.evaluated_samples} independent test samples.
          </p>
        </div>

        <button
          className="btn-secondary"
          disabled={evaluating}
          onClick={handleReEvaluate}
          style={{ padding: '8px 16px', fontSize: '0.8rem' }}
        >
          <RefreshCw size={14} className={evaluating ? 'pulse-glow' : ''} />
          {evaluating ? 'Evaluating...' : 'Re-Run Test Evaluation'}
        </button>
      </div>

      {/* KPI Metrics Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '16px'
      }}>
        <MetricCard
          title="Overall Accuracy"
          value={`${(perf.accuracy * 100).toFixed(2)}%`}
          subtext={`${perf.evaluated_samples} Test Samples`}
          icon={Award}
          color="#10B981"
        />
        <MetricCard
          title="Macro F1-Score"
          value={`${(perf.macro_f1 * 100).toFixed(2)}%`}
          subtext="Unweighted Class Mean"
          icon={BarChart2}
          color="#8B5CF6"
        />
        <MetricCard
          title="Weighted F1-Score"
          value={`${(perf.weighted_f1 * 100).toFixed(2)}%`}
          subtext="Support-Weighted Mean"
          icon={BarChart2}
          color="#06B6D4"
        />
        <MetricCard
          title="Macro Precision"
          value={`${(perf.macro_precision * 100).toFixed(2)}%`}
          subtext="Average Positive Predictive"
          icon={CheckCircle2}
          color="#EC4899"
        />
        <MetricCard
          title="Macro Recall"
          value={`${(perf.macro_recall * 100).toFixed(2)}%`}
          subtext="Average Sensitivity"
          icon={CheckCircle2}
          color="#F59E0B"
        />
        <MetricCard
          title="ROC-AUC (OVR)"
          value={perf.roc_auc ? perf.roc_auc.toFixed(4) : 'N/A'}
          subtext="Multiclass One-vs-Rest"
          icon={LineChartIcon}
          color="#38BDF8"
        />
      </div>

      {/* Training Curves: Accuracy vs Epoch & Loss vs Epoch */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Accuracy vs Epoch */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 4px 0' }}>Accuracy vs Epoch</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Training Accuracy vs Validation Accuracy trajectory
          </p>

          <div style={{ width: '100%', height: '240px' }}>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="epoch" stroke="var(--text-dim)" fontSize={10} />
                  <YAxis stroke="var(--text-dim)" fontSize={10} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#111827',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      fontSize: '0.78rem'
                    }}
                    formatter={(val) => [`${val}%`, '']}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '6px' }} />
                  <Line type="monotone" dataKey="trainAcc" name="Train Accuracy (%)" stroke="#10B981" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="valAcc" name="Val Accuracy (%)" stroke="#06B6D4" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                No training history curves available for current checkpoint.
              </div>
            )}
          </div>
        </div>

        {/* Loss vs Epoch */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 4px 0' }}>Loss vs Epoch</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Training Loss vs Validation Loss optimization
          </p>

          <div style={{ width: '100%', height: '240px' }}>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="epoch" stroke="var(--text-dim)" fontSize={10} />
                  <YAxis stroke="var(--text-dim)" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#111827',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '8px',
                      fontSize: '0.78rem'
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '6px' }} />
                  <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#EF4444" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#F59E0B" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                No loss history available.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Interactive Confusion Matrix Section */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>
              Interactive Confusion Matrix (6 × 6)
            </h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
              Green diagonal cells denote correct predictions. Red off-diagonal cells highlight misclassifications.
            </p>
          </div>
          <span className="badge badge-purple">Evaluated: {perf.evaluated_at}</span>
        </div>

        <ConfusionMatrix matrix={perf.confusion_matrix} classes={perf.classes} />
      </div>

      {/* Class-wise Breakdown Table & Chart */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
        {/* Class-wise Metrics Table */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 4px 0' }}>Class-Wise Performance Breakdown</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Precision, Recall, F1-Score, and Support counts for each commercial category
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Category</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Precision</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Recall</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>F1-Score</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Support</th>
                </tr>
              </thead>
              <tbody>
                {perf.class_wise_metrics.map((row) => (
                  <tr key={row.class_name} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '10px', fontWeight: 600, color: '#A78BFA' }}>{row.class_name}</td>
                    <td style={{ padding: '10px', color: '#10B981', fontWeight: 600 }}>{(row.precision * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px', color: '#67E8F9', fontWeight: 600 }}>{(row.recall * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px', color: '#FCD34D', fontWeight: 600 }}>{(row.f1_score * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px', color: 'var(--text-muted)' }}>{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Class-wise Comparative Chart */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 4px 0' }}>Category Metric Comparison</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Comparing Precision, Recall, and F1 across classes
          </p>

          <div style={{ width: '100%', height: '260px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classWiseData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="var(--text-dim)" fontSize={10} angle={-15} textAnchor="end" />
                <YAxis stroke="var(--text-dim)" fontSize={10} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    fontSize: '0.78rem'
                  }}
                  formatter={(val) => [`${val}%`, '']}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '8px' }} />
                <Bar dataKey="precision" name="Precision" fill="#10B981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="recall" name="Recall" fill="#06B6D4" radius={[3, 3, 0, 0]} />
                <Bar dataKey="f1" name="F1-Score" fill="#F59E0B" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
