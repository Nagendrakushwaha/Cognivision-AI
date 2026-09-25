import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  Sliders,
  TrendingUp,
  Activity
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid
} from 'recharts';
import { api } from '../services/api';

export default function ModelTraining({ onTrainingComplete }) {
  const [config, setConfig] = useState({
    model_name: 'mobilenet_v3',
    epochs: 5,
    batch_size: 32,
    learning_rate: 0.001,
    optimizer: 'Adam',
    validation_split: 0.2,
    random_seed: 42
  });

  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Fetch status on mount and poll while training
  useEffect(() => {
    let interval = null;

    async function checkStatus() {
      try {
        const s = await api.getTrainingStatus();
        setStatus(s);
        if (s.status === 'training') {
          if (!interval) {
            interval = setInterval(checkStatus, 1500);
          }
        } else {
          if (interval) {
            clearInterval(interval);
            interval = null;
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    checkStatus();
    return () => {
      if (interval) clearInterval(interval);
    };
  }, []);

  async function handleStartTraining(e) {
    e.preventDefault();
    setStarting(true);
    setErrorMsg('');

    try {
      await api.startTraining(config);
      // Poll every 1.5s
      const poll = setInterval(async () => {
        try {
          const s = await api.getTrainingStatus();
          setStatus(s);
          if (s.status !== 'training') {
            clearInterval(poll);
            setStarting(false);
            if (s.status === 'failed' && s.error_message) {
              setErrorMsg(`Training failed: ${s.error_message}`);
            }
            if (s.status === 'completed' && onTrainingComplete) {
              onTrainingComplete();
            }
          }
        } catch (pollErr) {
          console.error('Error polling training status:', pollErr);
        }
      }, 1500);
    } catch (err) {
      const msg = err.message ? (err.message.startsWith('Training failed:') ? err.message : `Training failed: ${err.message}`) : 'Training failed to start';
      setErrorMsg(msg);
      setStarting(false);
    }
  }

  const isTraining = status && status.status === 'training';
  const historyData = status && status.history ? status.history.map(h => ({
    epoch: `Epoch ${h.epoch}`,
    trainLoss: h.train_loss,
    valLoss: h.val_loss,
    trainAcc: (h.train_acc * 100).toFixed(1),
    valAcc: (h.val_acc * 100).toFixed(1),
    dur: h.duration_seconds
  })) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Model Training Pipeline</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Train lightweight CPU-optimized computer vision architectures on the 626 receipt training samples.
        </p>
      </div>

      {/* Main Grid: Config Form & Live Status */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 2fr', gap: '24px' }}>
        {/* Training Configuration Panel */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Sliders size={18} color="#8B5CF6" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Training Hyperparameters</h3>
          </div>

          <form onSubmit={handleStartTraining} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* Model Architecture */}
            <div>
              <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                Architecture
              </label>
              <select
                value={config.model_name}
                disabled={isTraining}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                style={{
                  width: '100%',
                  backgroundColor: '#1E293B',
                  color: '#FFFFFF',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '9px 12px',
                  fontSize: '0.85rem'
                }}
              >
                <option value="mobilenet_v3">MobileNetV3-Small (Transfer Learning • High Accuracy)</option>
                <option value="cogninet_cnn">CogniNet-CNN (Custom 4-Stage CNN • Fast Scratch)</option>
              </select>
            </div>

            {/* Epochs & Batch Size */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Epochs (1 - 20)
                </label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  disabled={isTraining}
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 1 })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Batch Size
                </label>
                <select
                  value={config.batch_size}
                  disabled={isTraining}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                >
                  <option value={16}>16 samples</option>
                  <option value={32}>32 samples (Recommended)</option>
                  <option value={64}>64 samples</option>
                </select>
              </div>
            </div>

            {/* Learning Rate & Optimizer */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Learning Rate
                </label>
                <select
                  value={config.learning_rate}
                  disabled={isTraining}
                  onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                >
                  <option value={0.0005}>0.0005 (Fine)</option>
                  <option value={0.001}>0.001 (Default)</option>
                  <option value={0.005}>0.005 (Fast)</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Optimizer
                </label>
                <select
                  value={config.optimizer}
                  disabled={isTraining}
                  onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                >
                  <option value="Adam">Adam</option>
                  <option value="AdamW">AdamW</option>
                  <option value="SGD">SGD (Momentum=0.9)</option>
                </select>
              </div>
            </div>

            {/* Validation Split & Random Seed */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Val Split (Ratio)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="0.3"
                  disabled={isTraining}
                  value={config.validation_split}
                  onChange={(e) => setConfig({ ...config, validation_split: parseFloat(e.target.value) })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                  Random Seed
                </label>
                <input
                  type="number"
                  disabled={isTraining}
                  value={config.random_seed}
                  onChange={(e) => setConfig({ ...config, random_seed: parseInt(e.target.value) || 42 })}
                  style={{
                    width: '100%',
                    backgroundColor: '#1E293B',
                    color: '#FFFFFF',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem'
                  }}
                />
              </div>
            </div>

            {errorMsg && (
              <div style={{
                padding: '10px 14px',
                borderRadius: '8px',
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#FCA5A5',
                fontSize: '0.78rem'
              }}>
                {errorMsg}
              </div>
            )}

            <button
              type="submit"
              disabled={isTraining || starting}
              className="btn-primary"
              style={{ width: '100%', marginTop: '6px' }}
            >
              {isTraining ? (
                <>
                  <Activity size={16} className="pulse-glow" /> Training in Progress...
                </>
              ) : (
                <>
                  <Play size={16} /> Start Local CPU Training
                </>
              )}
            </button>
          </form>
        </div>

        {/* Live Status & Progress Panel */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Execution Status</h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                Hardware Target: AMD Ryzen 5 5500U CPU • Cached Float32 Tensors
              </p>
            </div>
            <span className={`badge ${
              isTraining ? 'badge-amber' : status && status.status === 'completed' ? 'badge-emerald' : status && status.status === 'failed' ? 'badge-rose' : 'badge-purple'
            }`}>
              {status ? status.status.toUpperCase() : 'IDLE'}
            </span>
          </div>

          {status && status.status === 'failed' && status.error_message && (
            <div style={{
              padding: '10px 14px',
              borderRadius: '8px',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#FCA5A5',
              fontSize: '0.8rem'
            }}>
              <strong>Training Error:</strong> {status.error_message}
            </div>
          )}

          {/* Progress Bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '6px' }}>
              <span style={{ color: 'var(--text-muted)' }}>
                Epoch Progress: {status ? `${status.current_epoch} / ${status.total_epochs || config.epochs}` : '0 / 0'}
              </span>
              <span style={{ color: '#A78BFA', fontWeight: 700 }}>
                {status ? `${status.progress_percent || 0}%` : '0%'}
              </span>
            </div>
            <div style={{
              width: '100%',
              height: '8px',
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              borderRadius: '999px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${status ? status.progress_percent : 0}%`,
                background: 'linear-gradient(90deg, #8B5CF6 0%, #06B6D4 100%)',
                transition: 'width 0.4s ease'
              }} />
            </div>
          </div>

          {/* Current Epoch Metrics Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '12px'
          }}>
            <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Train Loss</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF', marginTop: '2px' }}>
                {status && status.current_train_loss !== null ? status.current_train_loss : '--'}
              </div>
            </div>
            <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Val Loss</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF', marginTop: '2px' }}>
                {status && status.current_val_loss !== null ? status.current_val_loss : '--'}
              </div>
            </div>
            <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Train Acc</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#10B981', marginTop: '2px' }}>
                {status && status.current_train_acc !== null ? `${(status.current_train_acc * 100).toFixed(1)}%` : '--'}
              </div>
            </div>
            <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Val Acc</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#67E8F9', marginTop: '2px' }}>
                {status && status.current_val_acc !== null ? `${(status.current_val_acc * 100).toFixed(1)}%` : '--'}
              </div>
            </div>
          </div>

          {/* Loss / Acc Curves */}
          <div style={{ width: '100%', height: '220px', marginTop: '4px' }}>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
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
                Start training to view live Loss and Accuracy curves.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Epochs History Table */}
      {status && status.history && status.history.length > 0 && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 14px 0' }}>Training Epoch History Log</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Epoch</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Train Loss</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Train Acc</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Val Loss</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Val Acc</th>
                  <th style={{ padding: '10px', color: 'var(--text-dim)' }}>Duration</th>
                </tr>
              </thead>
              <tbody>
                {status.history.map((h) => (
                  <tr key={h.epoch} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '10px', fontWeight: 600, color: '#A78BFA' }}>Epoch {h.epoch}</td>
                    <td style={{ padding: '10px', color: 'var(--text-main)' }}>{h.train_loss}</td>
                    <td style={{ padding: '10px', color: '#10B981', fontWeight: 600 }}>{(h.train_acc * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px', color: 'var(--text-main)' }}>{h.val_loss}</td>
                    <td style={{ padding: '10px', color: '#67E8F9', fontWeight: 600 }}>{(h.val_acc * 100).toFixed(1)}%</td>
                    <td style={{ padding: '10px', color: 'var(--text-dim)' }}>{h.duration_seconds}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
