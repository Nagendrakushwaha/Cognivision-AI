import React, { useState, useEffect } from 'react';
import {
  Database,
  Filter,
  Search,
  Eye,
  ChevronLeft,
  ChevronRight,
  X,
  FileText,
  Maximize2,
  Tag,
  Calendar,
  MapPin,
  DollarSign
} from 'lucide-react';
import { api } from '../services/api';

export default function DatasetExplorer({ onSelectSampleForPrediction }) {
  const [summary, setSummary] = useState(null);
  const [split, setSplit] = useState('train');
  const [category, setCategory] = useState('All');
  const [page, setPage] = useState(1);
  const [recordsData, setRecordsData] = useState({ records: [], total: 0, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  // Selected receipt for detailed modal inspection
  const [selectedReceipt, setSelectedReceipt] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    async function fetchSummary() {
      try {
        const sum = await api.getDatasetSummary();
        setSummary(sum);
      } catch (err) {
        console.error(err);
      }
    }
    fetchSummary();
  }, []);

  useEffect(() => {
    async function fetchRecords() {
      setLoading(true);
      try {
        const data = await api.getDatasetRecords(split, page, 16, category);
        setRecordsData(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchRecords();
  }, [split, category, page]);

  async function handleInspect(item) {
    setDetailLoading(true);
    try {
      const detail = await api.getReceiptDetail(item.split, item.idx);
      setSelectedReceipt(detail);
    } catch (err) {
      console.error(err);
    } finally {
      setDetailLoading(false);
    }
  }

  // Filter records locally by search query if typed
  const displayedRecords = recordsData.records.filter(r => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      r.key.toLowerCase().includes(q) ||
      (r.company && r.company.toLowerCase().includes(q)) ||
      (r.category && r.category.toLowerCase().includes(q))
    );
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', padding: '32px' }}>
      {/* Header & Meta Summary */}
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Dataset Explorer</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Inspect the official SROIE ICDAR benchmark parquet dataset records, image dimensions, extracted entities, and OCR tokens.
        </p>
      </div>

      {/* Dataset Schema & Health Summary Card */}
      {summary && (
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '16px'
          }}>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Total Records</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#FFFFFF' }}>{summary.total_records}</div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>626 Train / 361 Test</span>
            </div>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Features / Columns</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#A78BFA' }}>6 Columns</div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>image, key, size, entities, words, bboxes</span>
            </div>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Target Classification</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#67E8F9' }}>6 Categories</div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Commercial Merchant Types</span>
            </div>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Image Resolution</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#6EE7B7' }}>
                {summary.image_dimension_stats.mean_width.toFixed(0)} × {summary.image_dimension_stats.mean_height.toFixed(0)}
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Average W × H pixels</span>
            </div>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: 600 }}>Missing Values</span>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#10B981' }}>0.01%</div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>1 null address in train</span>
            </div>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="glass-panel" style={{
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '14px'
      }}>
        {/* Split Toggle */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={split === 'train' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.8rem' }}
            onClick={() => { setSplit('train'); setPage(1); }}
          >
            Train Set (626)
          </button>
          <button
            className={split === 'test' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.8rem' }}
            onClick={() => { setSplit('test'); setPage(1); }}
          >
            Test Set (361)
          </button>
        </div>

        {/* Category Selector */}
        {summary && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Category:</span>
            <select
              value={category}
              onChange={(e) => { setCategory(e.target.value); setPage(1); }}
              style={{
                backgroundColor: '#1E293B',
                color: '#FFFFFF',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '0.8rem',
                outline: 'none'
              }}
            >
              <option value="All">All Categories ({recordsData.total})</option>
              {summary.classes.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        )}

        {/* Search Input */}
        <div style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center'
        }}>
          <Search size={15} color="var(--text-dim)" style={{ position: 'absolute', left: '12px' }} />
          <input
            type="text"
            placeholder="Search key or merchant..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              backgroundColor: '#1E293B',
              color: '#FFFFFF',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '7px 12px 7px 34px',
              fontSize: '0.8rem',
              outline: 'none',
              width: '220px'
            }}
          />
        </div>
      </div>

      {/* Gallery Cards Grid */}
      {loading ? (
        <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading dataset records...
        </div>
      ) : displayedRecords.length === 0 ? (
        <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
          No records match the selected filters.
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '18px'
        }}>
          {displayedRecords.map((item) => (
            <div
              key={`${item.split}-${item.idx}`}
              className="glass-panel"
              style={{
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                transition: 'transform 0.2s ease, border-color 0.2s ease'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span className="code-font" style={{ fontSize: '0.78rem', color: '#A78BFA', fontWeight: 600 }}>
                    {item.key}
                  </span>
                  <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>
                    {item.split.toUpperCase()} #{item.idx}
                  </span>
                </div>

                <div style={{
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  color: '#FFFFFF',
                  marginBottom: '6px',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }} title={item.company || 'Unknown Merchant'}>
                  {item.company || 'Unknown Merchant'}
                </div>

                <div className="badge badge-cyan" style={{ fontSize: '0.68rem', marginBottom: '12px' }}>
                  {item.category}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Resolution:</span>
                    <span style={{ color: 'var(--text-main)' }}>{item.width} × {item.height} px</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>OCR Tokens:</span>
                    <span style={{ color: 'var(--text-main)' }}>{item.num_words} words</span>
                  </div>
                  {item.total && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Total Amount:</span>
                      <span style={{ color: '#10B981', fontWeight: 600 }}>RM {item.total}</span>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
                <button
                  className="btn-secondary"
                  style={{ flex: 1, padding: '7px', fontSize: '0.75rem' }}
                  onClick={() => handleInspect(item)}
                >
                  <Eye size={14} /> Inspect Details
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Showing page {recordsData.page} of {recordsData.total_pages} ({recordsData.total} total records)
        </span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn-secondary"
            disabled={page <= 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            style={{ padding: '6px 12px', fontSize: '0.78rem' }}
          >
            <ChevronLeft size={16} /> Prev
          </button>
          <button
            className="btn-secondary"
            disabled={page >= recordsData.total_pages}
            onClick={() => setPage(p => Math.min(recordsData.total_pages, p + 1))}
            style={{ padding: '6px 12px', fontSize: '0.78rem' }}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* Detailed Receipt Modal */}
      {selectedReceipt && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 999,
          padding: '20px'
        }}>
          <div className="glass-panel" style={{
            width: '900px',
            maxWidth: '95vw',
            maxHeight: '90vh',
            overflowY: 'auto',
            padding: '28px',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            gap: '20px'
          }}>
            {/* Modal Close Button */}
            <button
              onClick={() => setSelectedReceipt(null)}
              style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                background: 'rgba(255, 255, 255, 0.08)',
                border: 'none',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#FFFFFF',
                cursor: 'pointer'
              }}
            >
              <X size={18} />
            </button>

            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                <span className="badge badge-purple">{selectedReceipt.split.toUpperCase()} SAMPLE #{selectedReceipt.idx}</span>
                <span className="code-font" style={{ fontSize: '0.9rem', color: '#A78BFA', fontWeight: 700 }}>
                  {selectedReceipt.key}
                </span>
              </div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, margin: 0, color: '#FFFFFF' }}>
                {selectedReceipt.entities.company || 'Receipt Details'}
              </h3>
            </div>

            {/* Split View: Image Preview on Left, Extracted Entities on Right */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '24px' }}>
              {/* Receipt Image */}
              <div style={{
                borderRadius: '10px',
                overflow: 'hidden',
                backgroundColor: '#05070A',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                maxHeight: '440px'
              }}>
                {selectedReceipt.image_base64 ? (
                  <img
                    src={`data:image/jpeg;base64,${selectedReceipt.image_base64}`}
                    alt="Receipt Scanned Image"
                    style={{ maxHeight: '440px', width: 'auto', objectFit: 'contain' }}
                  />
                ) : (
                  <span style={{ color: 'var(--text-dim)' }}>Image not available</span>
                )}
              </div>

              {/* Extracted Entities & Metadata */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, margin: 0, color: 'var(--text-muted)' }}>
                  SROIE Ground Truth Key Entities
                </h4>

                <div className="glass-panel" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                    <Tag size={16} color="#8B5CF6" style={{ marginTop: '2px' }} />
                    <div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Company / Vendor</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#FFFFFF' }}>
                        {selectedReceipt.entities.company || 'Not extracted'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                    <Calendar size={16} color="#06B6D4" style={{ marginTop: '2px' }} />
                    <div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Date</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#FFFFFF' }}>
                        {selectedReceipt.entities.date || 'Not extracted'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                    <MapPin size={16} color="#A78BFA" style={{ marginTop: '2px' }} />
                    <div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Address</div>
                      <div style={{ fontSize: '0.8rem', color: '#D1D5DB' }}>
                        {selectedReceipt.entities.address || 'Not extracted'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                    <DollarSign size={16} color="#10B981" style={{ marginTop: '2px' }} />
                    <div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Total Amount</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#10B981' }}>
                        {selectedReceipt.entities.total ? `RM ${selectedReceipt.entities.total}` : 'Not extracted'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Technical Specs */}
                <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <div>Dimensions: <strong>{selectedReceipt.width} × {selectedReceipt.height} px</strong></div>
                  <div>OCR Words: <strong>{selectedReceipt.words_count}</strong></div>
                  <div>Format: <strong>JPEG (RGB)</strong></div>
                </div>

                {/* Tokens sample */}
                <div>
                  <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>
                    Sample OCR Words Detected:
                  </div>
                  <div style={{
                    maxHeight: '100px',
                    overflowY: 'auto',
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    padding: '8px',
                    borderRadius: '6px',
                    fontSize: '0.72rem',
                    color: '#9CA3AF',
                    lineHeight: 1.5
                  }}>
                    {selectedReceipt.tokens.slice(0, 30).map(t => t.text).join(' • ')}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
