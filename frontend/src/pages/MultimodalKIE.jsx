import React, { useState, useEffect } from 'react';
import {
  FileSearch,
  Tag,
  Calendar,
  MapPin,
  DollarSign,
  Layers,
  ChevronRight,
  Eye,
  CheckCircle2
} from 'lucide-react';
import { api } from '../services/api';

const ENTITY_COLORS = {
  company: { bg: 'rgba(59, 130, 246, 0.25)', border: '#3B82F6', text: '#93C5FD' },
  date: { bg: 'rgba(16, 185, 129, 0.25)', border: '#10B981', text: '#6EE7B7' },
  address: { bg: 'rgba(139, 92, 246, 0.25)', border: '#8B5CF6', text: '#C4B5FD' },
  total: { bg: 'rgba(245, 158, 11, 0.25)', border: '#F59E0B', text: '#FCD34D' },
  other: { bg: 'rgba(255, 255, 255, 0.05)', border: 'rgba(255, 255, 255, 0.2)', text: '#9CA3AF' }
};

export default function MultimodalKIE() {
  const [sampleIdx, setSampleIdx] = useState(0);
  const [split, setSplit] = useState('test');
  const [receipt, setReceipt] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeFilter, setActiveFilter] = useState('all'); // 'all', 'company', 'date', 'address', 'total'

  useEffect(() => {
    loadReceipt();
  }, [sampleIdx, split]);

  async function loadReceipt() {
    setLoading(true);
    try {
      const res = await api.getReceiptDetail(split, sampleIdx);
      setReceipt(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  const filteredTokens = receipt && receipt.tokens ? receipt.tokens.filter(t => {
    if (activeFilter === 'all') return true;
    return t.label === activeFilter;
  }) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Multimodal Key Information Extraction</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Inspect ground-truth OCR word tokens and 2D spatial bounding boxes aligned with structured key entities (Company, Date, Address, Total).
        </p>
      </div>

      {/* Control Selector Bar */}
      <div className="glass-panel" style={{
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '14px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Split:</span>
          <select
            value={split}
            onChange={(e) => setSplit(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#FFFFFF',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '6px 12px',
              fontSize: '0.8rem'
            }}
          >
            <option value="test">Test Set (361 receipts)</option>
            <option value="train">Train Set (626 receipts)</option>
          </select>

          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '8px' }}>Sample Index:</span>
          <input
            type="number"
            min="0"
            max={split === 'train' ? 625 : 360}
            value={sampleIdx}
            onChange={(e) => setSampleIdx(Math.max(0, parseInt(e.target.value) || 0))}
            style={{
              backgroundColor: '#1E293B',
              color: '#FFFFFF',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '6px 12px',
              fontSize: '0.8rem',
              width: '80px'
            }}
          />
        </div>

        {/* Entity Filter Badges */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {['all', 'company', 'date', 'address', 'total'].map(f => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '0.72rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                border: activeFilter === f ? '1px solid #8B5CF6' : '1px solid var(--border-subtle)',
                backgroundColor: activeFilter === f ? 'rgba(139, 92, 246, 0.2)' : 'rgba(255, 255, 255, 0.03)',
                color: activeFilter === f ? '#FFFFFF' : 'var(--text-muted)',
                cursor: 'pointer'
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading Multimodal Document Tokens...
        </div>
      ) : receipt ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.4fr', gap: '24px' }}>
          {/* Left: Scanned Receipt Preview */}
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Receipt Document Preview</h3>
                <span className="code-font" style={{ fontSize: '0.75rem', color: '#A78BFA' }}>{receipt.key}</span>
              </div>
              <span className="badge badge-purple">{receipt.width} × {receipt.height} px</span>
            </div>

            <div style={{
              backgroundColor: '#05070A',
              borderRadius: '8px',
              border: '1px solid var(--border-subtle)',
              height: '460px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden'
            }}>
              {receipt.image_base64 && (
                <img
                  src={`data:image/jpeg;base64,${receipt.image_base64}`}
                  alt="Receipt"
                  style={{ maxHeight: '460px', maxWidth: '100%', objectFit: 'contain' }}
                />
              )}
            </div>

            {/* Extracted Structured Key-Value Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                <span style={{ fontSize: '0.7rem', color: '#93C5FD', textTransform: 'uppercase', fontWeight: 600 }}>Company</span>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#FFFFFF', marginTop: '2px' }}>
                  {receipt.entities.company || 'N/A'}
                </div>
              </div>

              <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <span style={{ fontSize: '0.7rem', color: '#6EE7B7', textTransform: 'uppercase', fontWeight: 600 }}>Date</span>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#FFFFFF', marginTop: '2px' }}>
                  {receipt.entities.date || 'N/A'}
                </div>
              </div>

              <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                <span style={{ fontSize: '0.7rem', color: '#C4B5FD', textTransform: 'uppercase', fontWeight: 600 }}>Address</span>
                <div style={{ fontSize: '0.78rem', color: '#FFFFFF', marginTop: '2px', lineHeight: 1.3 }}>
                  {receipt.entities.address || 'N/A'}
                </div>
              </div>

              <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                <span style={{ fontSize: '0.7rem', color: '#FCD34D', textTransform: 'uppercase', fontWeight: 600 }}>Total</span>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#10B981', marginTop: '2px' }}>
                  {receipt.entities.total ? `RM ${receipt.entities.total}` : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Right: Detected OCR Tokens & Bounding Boxes Table */}
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Detected OCR Tokens</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Showing {filteredTokens.length} of {receipt.words_count} tokens
                </span>
              </div>
            </div>

            <div style={{ maxHeight: '560px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.78rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ padding: '8px', color: 'var(--text-dim)' }}>#</th>
                    <th style={{ padding: '8px', color: 'var(--text-dim)' }}>Detected Text</th>
                    <th style={{ padding: '8px', color: 'var(--text-dim)' }}>Entity Tag</th>
                    <th style={{ padding: '8px', color: 'var(--text-dim)' }}>BBox [x1, y1, x2, y2]</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTokens.map((t, idx) => {
                    const c = ENTITY_COLORS[t.label] || ENTITY_COLORS.other;
                    return (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                        <td style={{ padding: '8px', color: 'var(--text-dim)' }}>{idx + 1}</td>
                        <td style={{ padding: '8px', fontWeight: 600, color: '#FFFFFF' }}>{t.text}</td>
                        <td style={{ padding: '8px' }}>
                          <span style={{
                            padding: '3px 8px',
                            borderRadius: '4px',
                            fontSize: '0.68rem',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            backgroundColor: c.bg,
                            border: `1px solid ${c.border}`,
                            color: c.text
                          }}>
                            {t.label}
                          </span>
                        </td>
                        <td style={{ padding: '8px', color: 'var(--text-muted)', fontFamily: 'monospace', fontSize: '0.72rem' }}>
                          [{t.bbox.join(', ')}]
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
