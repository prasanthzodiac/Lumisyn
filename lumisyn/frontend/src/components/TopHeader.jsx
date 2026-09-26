import React from 'react';

export default function TopHeader({ title = "Dashboard", sessionData }) {
  return (
    <header className="top-header">
      {/* ── Left: Title ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
        <div style={{
          width: '3px', height: '16px',
          background: 'linear-gradient(180deg, var(--accent-cyan), var(--accent-violet))',
          borderRadius: '2px',
          flexShrink: 0,
        }} />
        <h2 style={{
          fontSize: '13px',
          fontWeight: '600',
          color: 'var(--text-primary)',
          letterSpacing: '-0.01em',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>{title}</h2>
        {sessionData && (
          <span className="badge badge--violet" style={{ fontSize: '9px', padding: '1.5px 6px', marginLeft: '6px', flexShrink: 0 }}>
            Active Dataset: {sessionData.filename}
          </span>
        )}
      </div>

      {/* ── Right: Actions ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
      </div>
    </header>
  );
}
