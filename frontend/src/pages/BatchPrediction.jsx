import React, { useState } from 'react';
import {
  Layers,
  Upload,
  Download,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Play,
  Clock
} from 'lucide-react';
import { api } from '../services/api';

export default function BatchPrediction() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [sampleBatchCount, setSampleBatchCount] = useState(10);
  const [loading, setLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  function handleFilesChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(Array.from(e.target.files));
      setErrorMsg('');
    }
  }

  async function handleRunBatchUpload() {
    if (selectedFiles.length === 0) return;
    setLoading(true);
    setErrorMsg('');

    try {
      const res = await api.predictBatch(selectedFiles);
      setBatchResult(res);
    } catch (err) {
      setErrorMsg(err.message || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleRunBatchTestSet() {
    setLoading(true);
    setErrorMsg('');
    setSelectedFiles([]);

    try {
      const res = await api.predictBatchTestSamples(sampleBatchCount);
      setBatchResult(res);
    } catch (err) {
      setErrorMsg(err.message || 'Batch test prediction failed');
    } finally {
      setLoading(false);
    }
  }

  function exportToCSV() {
    if (!batchResult || !batchResult.results || batchResult.results.length === 0) return;

    const headers = ['Filename', 'Predicted Class', 'Confidence (%)', 'Top 2 Alternative', 'Alternative Conf (%)'];
    const rows = batchResult.results.map(r => [
      `"${r.filename}"`,
      `"${r.predicted_class}"`,
      r.confidence_percentage,
      `"${r.top_2_class}"`,
      (r.top_2_confidence * 100).toFixed(1)
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `cognivision_batch_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Batch Prediction Engine</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Process multiple receipts simultaneously, view classification confidence, and export predictions to CSV.
        </p>
      </div>

      {/* Batch Control Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Multi-file Upload */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 8px 0' }}>Upload Multiple Files</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Select several receipt scans from your local storage to batch process
          </p>

          <label style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '24px 16px',
            border: '2px dashed var(--border-subtle)',
            borderRadius: '10px',
            backgroundColor: 'rgba(255, 255, 255, 0.02)',
            cursor: 'pointer',
            marginBottom: '16px'
          }}>
            <Upload size={20} color="#8B5CF6" />
            <span style={{ fontSize: '0.85rem', color: '#FFFFFF', fontWeight: 600 }}>
              {selectedFiles.length > 0 ? `${selectedFiles.length} files selected` : 'Select Multiple Images'}
            </span>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFilesChange}
              style={{ display: 'none' }}
            />
          </label>

          <button
            className="btn-primary"
            disabled={selectedFiles.length === 0 || loading}
            onClick={handleRunBatchUpload}
            style={{ width: '100%' }}
          >
            <Play size={15} /> Run Batch on {selectedFiles.length} Files
          </button>
        </div>

        {/* Batch Test Samples */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 8px 0' }}>Batch Test Set Evaluation</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Run batch inference across a sequence of receipts from the SROIE test set
          </p>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Number of Samples:</span>
            <select
              value={sampleBatchCount}
              onChange={(e) => setSampleBatchCount(parseInt(e.target.value))}
              style={{
                backgroundColor: '#1E293B',
                color: '#FFFFFF',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '8px 14px',
                fontSize: '0.82rem'
              }}
            >
              <option value={5}>5 Samples</option>
              <option value={10}>10 Samples</option>
              <option value={20}>20 Samples</option>
              <option value={30}>30 Samples</option>
            </select>
          </div>

          <button
            className="btn-secondary"
            disabled={loading}
            onClick={handleRunBatchTestSet}
            style={{ width: '100%', padding: '10px' }}
          >
            <Layers size={16} color="#06B6D4" /> Evaluate {sampleBatchCount} Test Receipts
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

      {/* Batch Results Table */}
      {batchResult && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>
                Batch Prediction Output ({batchResult.total_processed} items)
              </h3>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Total latency: {batchResult.inference_time_ms} ms • Model: {batchResult.model_used}
              </span>
            </div>

            <button className="btn-primary" onClick={exportToCSV} style={{ padding: '8px 16px', fontSize: '0.78rem' }}>
              <Download size={14} /> Export Results to CSV
            </button>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '12px', color: 'var(--text-dim)' }}>Sample Filename / Key</th>
                  <th style={{ padding: '12px', color: 'var(--text-dim)' }}>Predicted Class</th>
                  <th style={{ padding: '12px', color: 'var(--text-dim)' }}>Confidence</th>
                  <th style={{ padding: '12px', color: 'var(--text-dim)' }}>Secondary Alternative</th>
                </tr>
              </thead>
              <tbody>
                {batchResult.results.map((r, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '12px', fontWeight: 600, color: '#A78BFA' }}>
                      {r.filename}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                        {r.predicted_class}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '80px', height: '6px', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '999px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${r.confidence_percentage}%`,
                            backgroundColor: '#10B981'
                          }} />
                        </div>
                        <span style={{ color: '#10B981', fontWeight: 600 }}>{r.confidence_percentage}%</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>
                      {r.top_2_class} ({(r.top_2_confidence * 100).toFixed(1)}%)
                    </td>
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
