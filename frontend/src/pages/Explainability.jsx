import React, { useState } from 'react';
import {
  Sparkles,
  Layers,
  Upload,
  Eye,
  Info,
  Sliders,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { api } from '../services/api';

export default function Explainability() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleIdx, setSampleIdx] = useState(3); // Default to Restaurant Wan Sheng
  const [explainResult, setExplainResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay', 'heatmap', 'original', 'compare'

  const sampleOptions = [
    { label: 'Sample #3 (Restaurant Wan Sheng)', idx: 3 },
    { label: 'Sample #4 (Gardenia Bakeries)', idx: 4 },
    { label: 'Sample #2 (MR DIY Hardware)', idx: 2 },
    { label: 'Sample #0 (Book Ta Stationery)', idx: 0 },
    { label: 'Sample #1 (Indah Gift / Retail)', idx: 1 },
    { label: 'Sample #5 (99 Speedmart)', idx: 5 }
  ];

  async function handleExplainSample() {
    setLoading(true);
    setErrorMsg('');
    try {
      const res = await api.explainSample('test', sampleIdx);
      setExplainResult(res);
    } catch (err) {
      setErrorMsg(err.message || 'Explainability computation failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleExplainUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setLoading(true);
    setErrorMsg('');

    try {
      const res = await api.explainImage(file);
      setExplainResult(res);
    } catch (err) {
      setErrorMsg(err.message || 'Upload explainability computation failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Explainable AI (Grad-CAM)</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Inspect Gradient-weighted Class Activation Mapping (Grad-CAM) across convolutional feature layers to visualize
          the exact spatial regions driving the model's classifications.
        </p>
      </div>

      {/* Input Selection Bar */}
      <div className="glass-panel" style={{
        padding: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Choose Benchmark Sample:</span>
          <select
            value={sampleIdx}
            onChange={(e) => setSampleIdx(parseInt(e.target.value))}
            style={{
              backgroundColor: '#1E293B',
              color: '#FFFFFF',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '8px 14px',
              fontSize: '0.82rem'
            }}
          >
            {sampleOptions.map(opt => (
              <option key={opt.idx} value={opt.idx}>{opt.label}</option>
            ))}
          </select>
          <button
            className="btn-primary"
            disabled={loading}
            onClick={handleExplainSample}
            style={{ padding: '8px 16px', fontSize: '0.8rem' }}
          >
            <Sparkles size={14} /> {loading ? 'Computing CAM...' : 'Generate Grad-CAM'}
          </button>
        </div>

        <div>
          <label className="btn-secondary" style={{ padding: '8px 16px', fontSize: '0.8rem', cursor: 'pointer' }}>
            <Upload size={14} /> Upload Custom Receipt
            <input type="file" accept="image/*" onChange={handleExplainUpload} style={{ display: 'none' }} />
          </label>
        </div>
      </div>

      {errorMsg && (
        <div style={{
          padding: '12px 16px',
          borderRadius: '8px',
          backgroundColor: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#FCA5A5',
          fontSize: '0.8rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <AlertCircle size={16} /> {errorMsg}
        </div>
      )}

      {/* Main Inspection View */}
      {explainResult && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '24px' }}>
          {/* Visual Display */}
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Visual Activation Map</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Grad-CAM Feature Map on Target Layer
                </span>
              </div>

              {/* View Mode Toggle */}
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  className={viewMode === 'overlay' ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                  onClick={() => setViewMode('overlay')}
                >
                  Blended Overlay
                </button>
                <button
                  className={viewMode === 'heatmap' ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                  onClick={() => setViewMode('heatmap')}
                >
                  Heatmap Only
                </button>
                <button
                  className={viewMode === 'original' ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                  onClick={() => setViewMode('original')}
                >
                  Original
                </button>
              </div>
            </div>

            {/* Image Box */}
            <div style={{
              height: '420px',
              backgroundColor: '#05070A',
              borderRadius: '10px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden'
            }}>
              <img
                src={
                  viewMode === 'heatmap'
                    ? `data:image/jpeg;base64,${explainResult.heatmap_only_base64}`
                    : viewMode === 'original'
                    ? `data:image/jpeg;base64,${explainResult.original_image_base64}`
                    : `data:image/jpeg;base64,${explainResult.overlay_base64}`
                }
                alt="Grad-CAM Visualization"
                style={{ maxHeight: '420px', maxWidth: '100%', objectFit: 'contain' }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              <span>Low Activation (Blue / Transparent)</span>
              <div style={{ flex: 1, height: '6px', background: 'linear-gradient(90deg, blue, cyan, yellow, red)', borderRadius: '3px' }} />
              <span style={{ color: '#F87171' }}>High Activation (Red / Peak)</span>
            </div>
          </div>

          {/* Qualitative Insights & Regional Weights */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Prediction Summary */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>
                Classification Output
              </span>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#FFFFFF', margin: '4px 0 6px 0' }}>
                {explainResult.predicted_class}
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="badge badge-emerald">Confidence: {explainResult.confidence_percentage}%</span>
                <span className="badge badge-purple">{explainResult.method}</span>
              </div>
            </div>

            {/* Regional Focus Distribution */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h4 style={{ fontSize: '0.88rem', fontWeight: 700, margin: '0 0 12px 0' }}>
                Spatial Quadrant Activation Weight
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(explainResult.region_importance).map(([region, score]) => (
                  <div key={region}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{region}</span>
                      <strong style={{ color: '#A78BFA' }}>{score}%</strong>
                    </div>
                    <div style={{ width: '100%', height: '6px', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '999px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${score}%`,
                        background: 'linear-gradient(90deg, #8B5CF6 0%, #06B6D4 100%)'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Analytical Interpretation */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h4 style={{ fontSize: '0.88rem', fontWeight: 700, margin: '0 0 8px 0', color: '#67E8F9' }}>
                Interpretability Insights
              </h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', lineHeight: 1.6, margin: 0 }}>
                {explainResult.explanation_text}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Theory & Methodological Card */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <Info size={18} color="#8B5CF6" />
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>How Grad-CAM Operates on Documents</h3>
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
          Gradient-weighted Class Activation Mapping calculates the gradients of the score for target category $c$ with respect
          to the feature activation maps $A^k$ of the convolutional backbone's final stage. These gradients are pooled to generate
          importance weights $\alpha_k^c$, capturing the visual attention of the model. In receipt documents, Grad-CAM highlights
          whether the network pays attention to merchant headers, currency notation, table formats, or itemized line items.
        </p>
      </div>
    </div>
  );
}
