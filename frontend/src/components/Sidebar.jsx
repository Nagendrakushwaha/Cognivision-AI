import React from 'react';
import {
  LayoutDashboard,
  Database,
  BarChart3,
  Cpu,
  LineChart,
  ScanEye,
  Layers,
  Sparkles,
  FileSearch,
  Info,
  ChevronRight
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'dataset-explorer', label: 'Dataset Explorer', icon: Database },
  { id: 'data-analysis', label: 'Data Analysis', icon: BarChart3 },
  { id: 'model-training', label: 'Model Training', icon: Cpu },
  { id: 'model-performance', label: 'Model Performance', icon: LineChart },
  { id: 'predictions', label: 'Predictions', icon: ScanEye },
  { id: 'batch-prediction', label: 'Batch Prediction', icon: Layers },
  { id: 'explainability', label: 'Explainability', icon: Sparkles },
  { id: 'multimodal-kie', label: 'Multimodal KIE', icon: FileSearch },
  { id: 'about', label: 'About Project', icon: Info },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside style={{
      width: '260px',
      height: '100vh',
      position: 'sticky',
      top: 0,
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border-subtle)',
      padding: '24px 16px',
      userSelect: 'none',
      zIndex: 40
    }}>
      {/* Brand Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '0 8px 24px 8px',
        borderBottom: '1px solid var(--border-subtle)',
        marginBottom: '20px'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #8B5CF6 0%, #06B6D4 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 16px rgba(139, 92, 246, 0.4)'
        }}>
          <ScanEye size={22} color="#FFFFFF" />
        </div>
        <div>
          <h1 style={{
            fontSize: '1.1rem',
            fontWeight: 800,
            letterSpacing: '0.04em',
            margin: 0,
            lineHeight: 1.2
          }}>
            COGNIVISION <span style={{ color: 'var(--secondary)' }}>AI</span>
          </h1>
          <p style={{
            fontSize: '0.68rem',
            color: 'var(--text-muted)',
            fontWeight: 500,
            marginTop: '2px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            Document Vision Intelligence
          </p>
        </div>
      </div>

      {/* Navigation List */}
      <nav style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        flex: 1,
        overflowY: 'auto'
      }}>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 14px',
                borderRadius: '9px',
                fontSize: '0.85rem',
                fontWeight: isActive ? 600 : 500,
                color: isActive ? '#FFFFFF' : 'var(--text-muted)',
                backgroundColor: isActive ? 'rgba(139, 92, 246, 0.18)' : 'transparent',
                border: isActive ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.18s ease',
                textAlign: 'left',
                width: '100%'
              }}
            >
              <Icon size={18} color={isActive ? '#A78BFA' : 'var(--text-muted)'} />
              <span style={{ flex: 1 }}>{item.label}</span>
              {isActive && <ChevronRight size={14} color="#A78BFA" />}
            </button>
          );
        })}
      </nav>

      {/* Hardware / Environment Status */}
      <div style={{
        padding: '14px',
        borderRadius: '10px',
        backgroundColor: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid var(--border-subtle)',
        marginTop: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10B981' }} />
          <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            AMD Ryzen 5 5500U
          </span>
        </div>
        <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', margin: 0 }}>
          Local CPU Inference Mode • 16GB RAM
        </p>
      </div>
    </aside>
  );
}
