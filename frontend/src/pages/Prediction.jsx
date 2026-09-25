import React, { useState, useEffect } from 'react';
import {
  Upload,
  ScanEye,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  ArrowRight,
  Image as ImageIcon,
  Layers,
  ChevronDown
} from 'lucide-react';
import { api } from '../services/api';

export default function Prediction() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [sampleSplit, setSampleSplit] = useState('test');
  const [includeGradcam, setIncludeGradcam] = useState(true);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [showGradcamOverlay, setShowGradcamOverlay] = useState(false);

  // Quick test samples list
  const testSampleOptions = [
    { label: 'Sample #0 (Book Ta / Stationery)', idx: 0 },
    { label: 'Sample #1 (Indah Gift / Retail)', idx: 1 },
    { label: 'Sample #2 (MR DIY / Hardware)', idx: 2 },
    { label: 'Sample #3 (Restaurant Wan Sheng)', idx: 3 },
    { label: 'Sample #4 (Gardenia Bakery)', idx: 4 },
    { label: 'Sample #5 (99 Speedmart / Grocery)', idx: 5 }
  ];

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg('');
    }
  }

  async function handlePredictUpload() {
    if (!selectedFile) return;
    setLoading(true);
    setErrorMsg('');

    try {
      const res = await api.predictImage(selectedFile, includeGradcam);
      setResult(res);
      setShowGradcamOverlay(!!res.gradcam_base64);
    } catch (err) {
      setErrorMsg(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  async function handlePredictSample() {
    setLoading(true);
    setErrorMsg('');
    setSelectedFile(null);
    setPreviewUrl(null);

    try {
      const res = await api.predictSample(sampleSplit, sampleIdx);
      setResult(res);
      setShowGradcamOverlay(!!res.gradcam_base64);
    } catch (err) {
      setErrorMsg(err.message || 'Sample prediction failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Live Prediction System</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Upload any receipt image or test against authentic SROIE benchmark samples to get category classifications,
          confidence scores, and top-k distributions.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.4fr', gap: '24px' }}>
        {/* Left: Input Selection Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* File Upload Box */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 12px 0' }}>Upload Receipt Document</h3>
            
            <label style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '32px 20px',
              border: '2px dashed var(--border-subtle)',
              borderRadius: '12px',
              backgroundColor: 'rgba(255, 255, 255, 0.02)',
              cursor: 'pointer',
              transition: 'border-color 0.2s ease',
              marginBottom: '16px'
            }}>
              <Upload size={28} color="#8B5CF6" style={{ marginBottom: '10px' }} />
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#FFFFFF' }}>
                {selectedFile ? selectedFile.name : 'Choose an image or drag & drop'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                Supports JPEG, PNG, WEBP (scanned receipts)
              </span>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </label>

            {previewUrl && (
              <div style={{
                maxHeight: '180px',
                overflow: 'hidden',
                borderRadius: '8px',
                marginBottom: '16px',
                textAlign: 'center',
                backgroundColor: '#05070A'
              }}>
                <img src={previewUrl} alt="Preview" style={{ maxHeight: '180px', objectFit: 'contain' }} />
              </div>
            )}

            <button
              className="btn-primary"
              disabled={!selectedFile || loading}
              onClick={handlePredictUpload}
              style={{ width: '100%' }}
            >
              <ScanEye size={16} /> {loading ? 'Running Inference...' : 'Predict Uploaded Image'}
            </button>
          </div>

          {/* Quick Select from Dataset */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 8px 0' }}>Or Select from Actual Dataset</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 14px 0' }}>
              Instantly test inference against ground-truth SROIE receipt records
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <select
                value={sampleIdx}
                onChange={(e) => setSampleIdx(parseInt(e.target.value))}
                style={{
                  backgroundColor: '#1E293B',
                  color: '#FFFFFF',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '9px 12px',
                  fontSize: '0.82rem'
                }}
              >
                {testSampleOptions.map(opt => (
                  <option key={opt.idx} value={opt.idx}>{opt.label}</option>
                ))}
              </select>

              <button
                className="btn-secondary"
                disabled={loading}
                onClick={handlePredictSample}
                style={{ width: '100%', padding: '10px' }}
              >
                <Layers size={16} color="#06B6D4" /> Test Dataset Sample #{sampleIdx}
              </button>
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
        </div>

        {/* Right: Prediction Results Panel */}
        <div className="glass-panel" style={{ padding: '28px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>Prediction Outcome</h3>
              {result && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
                    <Clock size={11} /> {result.inference_time_ms} ms
                  </span>
                  <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                    {result.model_used}
                  </span>
                </div>
              )}
            </div>

            {result ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Predicted Class Card */}
                <div style={{
                  padding: '20px',
                  borderRadius: '12px',
                  background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(6, 182, 212, 0.1) 100%)',
                  border: '1px solid rgba(139, 92, 246, 0.4)'
                }}>
                  <span style={{ fontSize: '0.72rem', color: '#C4B5FD', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em' }}>
                    Predicted Category
                  </span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#FFFFFF', margin: '4px 0 10px 0' }}>
                    {result.predicted_class}
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Confidence Probability</span>
                      <strong style={{ color: '#10B981' }}>{result.confidence_percentage}%</strong>
                    </div>
                    <div style={{ width: '100%', height: '8px', backgroundColor: 'rgba(255, 255, 255, 0.1)', borderRadius: '999px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${result.confidence_percentage}%`,
                        background: 'linear-gradient(90deg, #10B981 0%, #06B6D4 100%)'
                      }} />
                    </div>
                  </div>
                </div>

                {/* Top-K Predictions List */}
                <div>
                  <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px' }}>
                    Top Category Probabilities
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {result.top_k.map((item, idx) => (
                      <div key={item.class_name} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                          <span style={{ color: idx === 0 ? '#FFFFFF' : 'var(--text-muted)', fontWeight: idx === 0 ? 600 : 400 }}>
                            {item.class_name}
                          </span>
                          <span style={{ color: idx === 0 ? '#10B981' : 'var(--text-dim)', fontWeight: 600 }}>
                            {item.percentage}%
                          </span>
                        </div>
                        <div style={{ width: '100%', height: '5px', backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: '999px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${item.percentage}%`,
                            backgroundColor: idx === 0 ? '#8B5CF6' : 'rgba(255, 255, 255, 0.2)'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Visual Preview / GradCAM Switch */}
                {(result.image_preview_base64 || result.gradcam_base64) && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                        Visual Inspection
                      </span>
                      {result.gradcam_base64 && (
                        <div style={{ display: 'flex', gap: '6px' }}>
                          <button
                            onClick={() => setShowGradcamOverlay(false)}
                            className={!showGradcamOverlay ? 'btn-primary' : 'btn-secondary'}
                            style={{ padding: '4px 10px', fontSize: '0.7rem' }}
                          >
                            Original
                          </button>
                          <button
                            onClick={() => setShowGradcamOverlay(true)}
                            className={showGradcamOverlay ? 'btn-primary' : 'btn-secondary'}
                            style={{ padding: '4px 10px', fontSize: '0.7rem' }}
                          >
                            <Sparkles size={11} /> Grad-CAM Heatmap
                          </button>
                        </div>
                      )}
                    </div>

                    <div style={{
                      height: '240px',
                      backgroundColor: '#05070A',
                      borderRadius: '8px',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      overflow: 'hidden'
                    }}>
                      <img
                        src={showGradcamOverlay && result.gradcam_base64
                          ? `data:image/jpeg;base64,${result.gradcam_base64}`
                          : `data:image/jpeg;base64,${result.image_preview_base64}`
                        }
                        alt="Prediction Visual"
                        style={{ maxHeight: '240px', maxWidth: '100%', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{
                height: '320px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-dim)',
                textAlign: 'center'
              }}>
                <ScanEye size={48} strokeWidth={1.2} style={{ marginBottom: '14px', opacity: 0.5 }} />
                <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>No Prediction Generated Yet</span>
                <span style={{ fontSize: '0.75rem', maxWidth: '300px', marginTop: '6px' }}>
                  Upload a receipt image or select a sample from the dataset to run deep learning inference.
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
