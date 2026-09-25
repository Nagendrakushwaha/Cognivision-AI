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
  Activity,
  ArrowRight,
  Sparkles
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
  const [internalViewTab, setInternalViewTab] = useState('architecture');

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

      {/* Inside the Model: Architecture & Training Dynamics */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Cpu size={20} color="#06B6D4" />
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                Neural Network Internals: {config.model_name === 'cogninet_cnn' ? 'CogniNet-CNN' : 'MobileNetV3-Small'}
              </h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '2px 0 0 0' }}>
                How receipt visual features propagate forward, compute loss, and update gradients during training
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px', backgroundColor: 'rgba(255, 255, 255, 0.04)', padding: '4px', borderRadius: '8px' }}>
            <button
              onClick={() => setInternalViewTab('architecture')}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                border: 'none',
                cursor: 'pointer',
                backgroundColor: internalViewTab === 'architecture' ? '#8B5CF6' : 'transparent',
                color: internalViewTab === 'architecture' ? '#FFFFFF' : 'var(--text-dim)',
                transition: 'all 0.2s ease'
              }}
            >
              Layer-by-Layer Architecture
            </button>
            <button
              onClick={() => setInternalViewTab('pipeline')}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                border: 'none',
                cursor: 'pointer',
                backgroundColor: internalViewTab === 'pipeline' ? '#8B5CF6' : 'transparent',
                color: internalViewTab === 'pipeline' ? '#FFFFFF' : 'var(--text-dim)',
                transition: 'all 0.2s ease'
              }}
            >
              4-Phase Training Loop
            </button>
          </div>
        </div>

        {internalViewTab === 'architecture' ? (
          <div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: '14px',
              marginBottom: '16px'
            }}>
              {config.model_name === 'cogninet_cnn' ? (
                <>
                  {/* Stage 1 */}
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge badge-purple" style={{ fontSize: '0.68rem' }}>Stage 1 • Conv2D</span>
                      <span style={{ fontSize: '0.72rem', color: '#A78BFA', fontWeight: 600 }}>[B, 32, 112, 112]</span>
                    </div>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Edge & Stroke Extraction</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      32 filters (3x3), BatchNorm2d, ReLU, MaxPool(2x2). Detects raw receipt paper borders, high-contrast ink edges, and character stroke lines.
                    </p>
                  </div>

                  {/* Stage 2 */}
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge badge-cyan" style={{ fontSize: '0.68rem' }}>Stage 2 • Conv2D</span>
                      <span style={{ fontSize: '0.72rem', color: '#67E8F9', fontWeight: 600 }}>[B, 64, 56, 56]</span>
                    </div>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Word & Numerical Blocks</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      64 filters (3x3), BatchNorm2d, ReLU, MaxPool(2x2). Synthesizes strokes into character groups, currency signs ($/RM), item prices, and date stamps.
                    </p>
                  </div>

                  {/* Stage 3 */}
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge badge-emerald" style={{ fontSize: '0.68rem' }}>Stage 3 • Conv2D</span>
                      <span style={{ fontSize: '0.72rem', color: '#6EE7B7', fontWeight: 600 }}>[B, 128, 28, 28]</span>
                    </div>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Spatial Layout Geometry</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      128 filters (3x3), BatchNorm2d, ReLU, MaxPool(2x2). Captures multi-line tabular receipt structure: company headers, address blocks, itemized lists.
                    </p>
                  </div>

                  {/* Stage 4 */}
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge badge-amber" style={{ fontSize: '0.68rem' }}>Stage 4 • Conv2D (Grad-CAM)</span>
                      <span style={{ fontSize: '0.72rem', color: '#FCD34D', fontWeight: 600 }}>[B, 256, 14, 14]</span>
                    </div>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Semantic Category Cues</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      256 filters (3x3), BatchNorm2d, ReLU, MaxPool(2x2). High-level merchant category signatures (e.g. restaurant tables, grocery barcodes, fuel headers).
                    </p>
                  </div>

                  {/* Classifier Head */}
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(244, 63, 94, 0.3)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge" style={{ backgroundColor: 'rgba(244, 63, 94, 0.2)', color: '#FDA4AF', fontSize: '0.68rem' }}>GAP + Linear</span>
                      <span style={{ fontSize: '0.72rem', color: '#FDA4AF', fontWeight: 600 }}>[B, 6] Logits</span>
                    </div>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Category Classification Head</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      AdaptiveAvgPool2d(1,1) collapses spatial grids into a 256-D vector. Dropout(0.3) + Linear(256→6) produces 6 category logits for CrossEntropyLoss.
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                    <span className="badge badge-purple" style={{ fontSize: '0.68rem', marginBottom: '8px', display: 'inline-block' }}>Input & Initial Conv</span>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Hard-Swish Feature Entry</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      Conv2D 3→16 with stride 2. Receptive field downsamples raw receipt to [B, 16, 112, 112] with hardware-friendly Hard-Swish non-linearities.
                    </p>
                  </div>

                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                    <span className="badge badge-cyan" style={{ fontSize: '0.68rem', marginBottom: '8px', display: 'inline-block' }}>11 Inverted Residual Blocks</span>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Depthwise Separable + SE Attention</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      Depthwise 3x3/5x5 convolutions combined with Squeeze-and-Excitation (SE) channel-attention blocks, extracting receipt patterns efficiently on CPU.
                    </p>
                  </div>

                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                    <span className="badge badge-emerald" style={{ fontSize: '0.68rem', marginBottom: '8px', display: 'inline-block' }}>Target Layer (Grad-CAM)</span>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Final Conv Feature Space</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      Final feature block output [B, 576, 7, 7]. Visual gradients during inference are backpropagated directly to this layer to compute Grad-CAM heatmaps.
                    </p>
                  </div>

                  <div style={{ padding: '16px', borderRadius: '10px', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                    <span className="badge badge-amber" style={{ fontSize: '0.68rem', marginBottom: '8px', display: 'inline-block' }}>Dense Head • 6 Classes</span>
                    <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 4px 0', color: '#FFFFFF' }}>Linear(576→1024→6)</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                      Global Average Pooling collapses spatial grid to 576-D vector. Two-stage projection with Dropout(0.2) outputs 6 merchant class confidence logits.
                    </p>
                  </div>
                </>
              )}
            </div>

            <div style={{ padding: '12px 16px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>
                <strong>Tensor Flow:</strong> [Batch, 3, 224, 224] → 4 Conv Stages → Global Avg Pool → [Batch, 256] → Linear → [Batch, 6] Logits
              </span>
              <span style={{ color: '#10B981', fontWeight: 600 }}>Zero GPU Required • Float32 CPU Inference</span>
            </div>
          </div>
        ) : (
          <div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '14px',
              marginBottom: '16px'
            }}>
              {/* Phase 1 */}
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                backgroundColor: isTraining ? 'rgba(139, 92, 246, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                border: isTraining ? '1px solid #8B5CF6' : '1px solid var(--border-subtle)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                  <span className="badge badge-purple" style={{ fontSize: '0.68rem' }}>Phase 1</span>
                  {isTraining && <Activity size={12} className="pulse-glow" color="#8B5CF6" />}
                </div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 6px 0', color: '#FFFFFF' }}>Forward Propagation</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                  Batch of 32 receipt tensors passes through all convolution and pooling layers. Each layer computes affine transform <code style={{ color: '#A78BFA' }}>Z = W·X + b</code> and ReLU activation, emitting raw logits <code style={{ color: '#A78BFA' }}>[32, 6]</code>.
                </p>
              </div>

              {/* Phase 2 */}
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                backgroundColor: isTraining ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                border: isTraining ? '1px solid #06B6D4' : '1px solid var(--border-subtle)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                  <span className="badge badge-cyan" style={{ fontSize: '0.68rem' }}>Phase 2</span>
                  {isTraining && <Activity size={12} className="pulse-glow" color="#06B6D4" />}
                </div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 6px 0', color: '#FFFFFF' }}>Loss Function Evaluation</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                  Cross-Entropy computes log-loss: <code style={{ color: '#67E8F9' }}>L = -log(e^(z_k) / Σ e^(z_j))</code>. Penalizes confident incorrect predictions and produces scalar loss value for the mini-batch.
                </p>
              </div>

              {/* Phase 3 */}
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                backgroundColor: isTraining ? 'rgba(16, 185, 129, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                border: isTraining ? '1px solid #10B981' : '1px solid var(--border-subtle)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                  <span className="badge badge-emerald" style={{ fontSize: '0.68rem' }}>Phase 3</span>
                  {isTraining && <Activity size={12} className="pulse-glow" color="#10B981" />}
                </div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 6px 0', color: '#FFFFFF' }}>Autograd Backpropagation</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                  PyTorch Autograd traverses the dynamic computational graph backward using the multivariable chain rule: <code style={{ color: '#6EE7B7' }}>∂L/∂W = ∂L/∂A · ∂A/∂Z · ∂Z/∂W</code>, computing exact weight gradients.
                </p>
              </div>

              {/* Phase 4 */}
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                backgroundColor: isTraining ? 'rgba(245, 158, 11, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                border: isTraining ? '1px solid #F59E0B' : '1px solid var(--border-subtle)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                  <span className="badge badge-amber" style={{ fontSize: '0.68rem' }}>Phase 4</span>
                  {isTraining && <Activity size={12} className="pulse-glow" color="#F59E0B" />}
                </div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, margin: '0 0 6px 0', color: '#FFFFFF' }}>{config.optimizer} Parameter Update</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0, lineHeight: 1.4 }}>
                  Adaptive moment estimation computes moving averages of gradients <code style={{ color: '#FCD34D' }}>m_t</code> and squared gradients <code style={{ color: '#FCD34D' }}>v_t</code>. Updates all weights: <code style={{ color: '#FCD34D' }}>W ← W - η·m̂/(√v̂ + ε)</code>.
                </p>
              </div>
            </div>

            <div style={{ padding: '12px 16px', borderRadius: '8px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>
                <strong>Optimization Step:</strong> loss.backward() computes tensor gradients → optimizer.step() updates weights → optimizer.zero_grad() resets buffers for next batch.
              </span>
              <span style={{ color: '#A78BFA', fontWeight: 600 }}>
                {status ? `Current Epoch Loss: ${status.current_train_loss || '--'}` : 'Ready'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
