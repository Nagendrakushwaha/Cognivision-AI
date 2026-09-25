import React from 'react';

export default function MetricCard({ title, value, subtext, icon: Icon, color = 'var(--primary)' }) {
  return (
    <div className="glass-panel" style={{
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background Glow */}
      <div style={{
        position: 'absolute',
        top: '-15px',
        right: '-15px',
        width: '70px',
        height: '70px',
        borderRadius: '50%',
        background: color,
        opacity: 0.12,
        filter: 'blur(20px)',
        pointerEvents: 'none'
      }} />

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {title}
        </span>
        {Icon && (
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '9px',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Icon size={18} color={color} />
          </div>
        )}
      </div>

      <div>
        <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em', lineHeight: 1.1 }}>
          {value}
        </div>
        {subtext && (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '6px' }}>
            {subtext}
          </div>
        )}
      </div>
    </div>
  );
}
