import React from 'react';
import { Activity, Database, CheckCircle2, Cpu } from 'lucide-react';

export default function Navbar({ activeTitle, isBackendHealthy = true, modelTrained = false, activeModelName = null }) {
  const formatModelName = (name) => {
    if (!name) return 'MobileNetV3';
    if (name.toLowerCase().includes('cogni')) return 'CogniNet-CNN';
    if (name.toLowerCase().includes('mobile')) return 'MobileNetV3';
    return name;
  };
  const displayModel = formatModelName(activeModelName);

  return (
    <header style={{
      height: '68px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 32px',
      borderBottom: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(10, 13, 20, 0.8)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 30
    }}>
      <div>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: '#F3F4F6' }}>
          {activeTitle}
        </h2>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Cognivision AI Platform • SROIE Benchmark Dataset
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        {/* Dataset Badge */}
        <div className="badge badge-purple" style={{ padding: '6px 12px' }}>
          <Database size={13} />
          <span>987 Records (626 Train / 361 Test)</span>
        </div>

        {/* Model Status Badge */}
        <div className={`badge ${modelTrained ? 'badge-emerald' : 'badge-amber'}`} style={{ padding: '6px 12px' }}>
          <CheckCircle2 size={13} />
          <span>{modelTrained ? `Model Active: ${displayModel}` : 'Model: Untrained'}</span>
        </div>

        {/* Python Version & Backend Status */}
        <div className="badge badge-cyan" style={{ padding: '6px 12px' }}>
          <Activity size={13} />
          <span>FastAPI • Python 3.13</span>
        </div>
      </div>
    </header>
  );
}
