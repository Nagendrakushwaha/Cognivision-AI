import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area
} from 'recharts';
import { BarChart3, PieChart as PieIcon, Sliders, CheckCircle2 } from 'lucide-react';
import { api } from '../services/api';

const COLORS = ['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#6366F1'];

export default function DataAnalysis() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const sum = await api.getDatasetSummary();
        setSummary(sum);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading Analytics...</div>;
  }

  // 1. Class Distribution Data
  const classDistData = summary ? summary.classes.map(cls => ({
    name: cls,
    train: summary.class_distribution_train[cls] || 0,
    test: summary.class_distribution_test[cls] || 0,
    total: (summary.class_distribution_train[cls] || 0) + (summary.class_distribution_test[cls] || 0)
  })) : [];

  // 2. Train vs Test Split Pie Data
  const splitPieData = [
    { name: 'Train Set', value: summary ? summary.train_records : 626, color: '#8B5CF6' },
    { name: 'Test Set', value: summary ? summary.test_records : 361, color: '#06B6D4' }
  ];

  // 3. Word Count Distribution Ranges
  const wordRangesData = [
    { range: '18 - 35 words', count: 98, note: 'Concise receipts / slips' },
    { range: '36 - 50 words', count: 242, note: 'Standard retail receipts' },
    { range: '51 - 70 words', count: 185, note: 'Multi-item supermarket bills' },
    { range: '71 - 100 words', count: 74, note: 'Long detailed invoices' },
    { range: '101 - 153 words', count: 27, note: 'Full-page itemized statements' }
  ];

  // 4. Missing Values Data
  const missingData = [
    { field: 'image', complete: 987, missing: 0, percentage: 100 },
    { field: 'key', complete: 987, missing: 0, percentage: 100 },
    { field: 'image_size', complete: 987, missing: 0, percentage: 100 },
    { field: 'entities.company', complete: 987, missing: 0, percentage: 100 },
    { field: 'entities.date', complete: 987, missing: 0, percentage: 100 },
    { field: 'entities.address', complete: 986, missing: 1, percentage: 99.9 },
    { field: 'entities.total', complete: 986, missing: 1, percentage: 99.9 },
    { field: 'words', complete: 987, missing: 0, percentage: 100 },
    { field: 'bboxes', complete: 987, missing: 0, percentage: 100 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', padding: '32px' }}>
      <div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, margin: '0 0 6px 0' }}>Data Analysis & Exploration</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: 0 }}>
          Comprehensive statistical profiling, class balance, document geometry, and OCR feature distributions.
        </p>
      </div>

      {/* Row 1: Class Distribution & Dataset Balance */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 6px 0' }}>
            Class Balance: Train vs Test Set
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Verification of consistent class proportions between training (63.4%) and testing (36.6%) splits.
          </p>

          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classDistData} layout="vertical" margin={{ top: 10, right: 30, left: 40, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis type="number" stroke="var(--text-dim)" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="var(--text-dim)" fontSize={11} width={130} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    fontSize: '0.78rem'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '8px' }} />
                <Bar dataKey="train" name="Train Samples" fill="#8B5CF6" radius={[0, 4, 4, 0]} />
                <Bar dataKey="test" name="Test Samples" fill="#06B6D4" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Dataset Split Ratio */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 6px 0', alignSelf: 'flex-start' }}>
            Train / Test Ratio
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0', alignSelf: 'flex-start' }}>
            63.4% Training • 36.6% Independent Evaluation
          </p>

          <div style={{ width: '100%', height: '200px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={splitPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {splitPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    fontSize: '0.78rem'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#8B5CF6' }} />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Train: 626</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#06B6D4' }} />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Test: 361</span>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: OCR Token Counts & Image Dimensions */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
        {/* Words / OCR tokens distribution */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 6px 0' }}>
            OCR Token Density Distribution
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Mean: 53.7 words per receipt • Min: 18 • Max: 153 tokens
          </p>

          <div style={{ width: '100%', height: '240px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={wordRangesData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                <defs>
                  <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="range" stroke="var(--text-dim)" fontSize={10} />
                <YAxis stroke="var(--text-dim)" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '8px',
                    fontSize: '0.78rem'
                  }}
                />
                <Area type="monotone" dataKey="count" name="Receipt Count" stroke="#10B981" fillOpacity={1} fill="url(#tokenGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Image Dimensions Summary */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 6px 0' }}>
            Document Geometry & Resolution
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
            Original scanned resolution stats across dataset
          </p>

          {summary && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Average Width:</span>
                <strong style={{ fontSize: '0.85rem', color: '#FFFFFF' }}>{summary.image_dimension_stats.mean_width.toFixed(1)} px</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Average Height:</span>
                <strong style={{ fontSize: '0.85rem', color: '#FFFFFF' }}>{summary.image_dimension_stats.mean_height.toFixed(1)} px</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Max Scanned Resolution:</span>
                <strong style={{ fontSize: '0.85rem', color: '#A78BFA' }}>{summary.image_dimension_stats.max_width} × {summary.image_dimension_stats.max_height} px</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Min Scanned Resolution:</span>
                <strong style={{ fontSize: '0.85rem', color: '#67E8F9' }}>{summary.image_dimension_stats.min_width} × {summary.image_dimension_stats.min_height} px</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Standard Aspect Ratio:</span>
                <strong style={{ fontSize: '0.85rem', color: '#10B981' }}>1 : 1.77 (Vertical Receipt Format)</strong>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Row 3: Data Quality & Missing Value Analysis */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: '0 0 6px 0' }}>
          Data Quality & Missing Values Audit
        </h3>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 16px 0' }}>
          Verification of schema integrity across all 6 primary and nested parquet fields
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Field Name</th>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Total Records</th>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Valid / Present</th>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Missing</th>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Completeness Rate</th>
                <th style={{ padding: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {missingData.map((row) => (
                <tr key={row.field} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <td style={{ padding: '10px', fontWeight: 600, color: '#A78BFA' }}>{row.field}</td>
                  <td style={{ padding: '10px', color: 'var(--text-main)' }}>987</td>
                  <td style={{ padding: '10px', color: '#10B981', fontWeight: 600 }}>{row.complete}</td>
                  <td style={{ padding: '10px', color: row.missing > 0 ? '#F59E0B' : 'var(--text-dim)' }}>{row.missing}</td>
                  <td style={{ padding: '10px', color: 'var(--text-main)' }}>{row.percentage}%</td>
                  <td style={{ padding: '10px' }}>
                    <span className={`badge ${row.missing === 0 ? 'badge-emerald' : 'badge-amber'}`} style={{ fontSize: '0.65rem' }}>
                      {row.missing === 0 ? 'Optimal' : 'Minor Null'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
