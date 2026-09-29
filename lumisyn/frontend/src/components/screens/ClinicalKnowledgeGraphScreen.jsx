import React, { useState, useEffect } from 'react';
import { GitBranch, ExternalLink, Info } from 'lucide-react';
import { apiUrl } from '../../api/client';

export default function ClinicalKnowledgeGraphScreen({ sessionData }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (sessionData?.session_id) {
      setLoading(true);
      fetch(apiUrl(`/api/graph/${sessionData.session_id}`))
        .then(res => res.json())
        .then(data => {
          setGraphData(data);
          if (data.nodes && data.nodes.length > 0) {
            setSelectedNode(data.nodes[0]);
          }
        })
        .catch(err => console.error("Graph fetch error:", err))
        .finally(() => setLoading(false));
    }
  }, [sessionData?.session_id]);

  const rawCols = sessionData?.columns || [];
  const columns = rawCols.map(c => (typeof c === 'object' && c !== null ? (c.name || String(c)) : String(c)));
  const target = String(sessionData?.suggested_target || 'Target Attribute');

  const staticPositions = [
    { top: '8%',   left: '50%', ml: '-40px', color: 'var(--accent-cyan)', bg: 'rgba(0,212,255,0.12)', border: 'rgba(0,212,255,0.35)' },
    { top: '25%',  left: '12%',  color: 'var(--accent-amber)', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.3)' },
    { top: '25%',  right: '12%', color: 'var(--accent-purple)', bg: 'rgba(168,85,247,0.1)', border: 'rgba(168,85,247,0.3)' },
    { bottom: '25%', left: '12%', color: 'var(--accent-emerald)', bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.3)' },
    { bottom: '25%', right: '12%', color: 'var(--accent-rose)', bg: 'rgba(244,63,94,0.1)', border: 'rgba(244,63,94,0.3)' },
    { bottom: '8%', left: '32%', color: 'var(--text-secondary)', bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.2)' },
    { bottom: '8%', right: '32%', color: 'var(--text-secondary)', bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.2)' },
  ];

  const nodesToUse = graphData?.nodes?.length > 0
    ? graphData.nodes.slice(0, 7)
    : columns.slice(0, 7).map(c => ({ name: c, sensitivity: 0.5, missing_pct: 0.0 }));

  const graphNodes = nodesToUse.map((n, i) => {
    const pos = staticPositions[i] || staticPositions[0];
    const name = n.name || n.id || `Feature_${i}`;
    const sens = n.sensitivity || 0.5;
    return {
      ...n,
      name,
      sensitivity: sens,
      ...pos
    };
  });

  const activeNode = selectedNode || graphNodes[0] || { name: target, sensitivity: 0.5 };
  const isCategorical = sessionData?.categorical_columns?.includes(activeNode.name);

  if (!sessionData?.session_id) {
    return <div className="screen-enter"><div className="panel" style={{ padding: '30px', textAlign: 'center' }}><Info size={26} color="var(--text-muted)" style={{ marginBottom: '8px' }} /><h3 style={{ fontSize: '15px' }}>No dataset selected</h3><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>The Stage 3 and Stage 3b graph will appear after dataset upload and graph analysis.</p></div></div>;
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr', gap: '16px' }} className="screen-enter">

      {/* ─── Graph Canvas ─── */}
      <div className="panel" style={{ padding: '20px', minHeight: '520px', position: 'relative', overflow: 'hidden' }}>
        {loading && <div style={{ color: 'var(--text-muted)', fontSize: '10px', marginBottom: '8px' }}>Loading measured graph statistics…</div>}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <div className="panel-title" style={{ margin: 0 }}>
            <GitBranch size={16} color="var(--accent-cyan)" />
            Multi-Relational Clinical Knowledge Graph (DCGB Stage 3 & 3b)
          </div>
          <span className="badge badge--cyan" style={{ fontSize: '10px' }}>
            {graphData ? `${graphData.node_count} Nodes | ${graphData.edge_count} Edges` : 'MI Fused Graph'}
          </span>
        </div>

        {/* SVG connecting lines */}
        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.75 }}>
          <defs>
            <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="3" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="var(--text-muted)" />
            </marker>
          </defs>
          {[['50%','50%','50%','14%'], ['50%','50%','18%','30%'], ['50%','50%','82%','30%'],
            ['50%','50%','18%','70%'], ['50%','50%','82%','70%']].map(([x1,y1,x2,y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="5 4"
              markerEnd="url(#arrowhead)" />
          ))}
        </svg>

        {/* Center Target Node */}
        <div style={{
          position: 'absolute',
          top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '105px', height: '105px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.25))',
          border: '3px solid var(--accent-violet)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 16px rgba(99,102,241,0.25)',
          zIndex: 10,
          textAlign: 'center',
          padding: '6px'
        }}>
          <span style={{ fontSize: '11px', fontWeight: '800', color: 'var(--text-primary)', wordBreak: 'break-word', lineHeight: 1.2 }}>{target}</span>
          <span style={{ fontSize: '9px', color: 'var(--accent-violet)', fontWeight: '700', marginTop: '2px' }}>Target Variable</span>
        </div>

        {/* Feature Nodes */}
        {graphNodes.map(n => (
          <button key={n.name} onClick={() => setSelectedNode(n)}
            style={{
              position: 'absolute',
              top: n.top, left: n.left, right: n.right, bottom: n.bottom,
              marginLeft: n.ml || 0,
              padding: '8px 12px',
              borderRadius: 'var(--radius-md)',
              background: activeNode.name === n.name ? n.bg : 'var(--bg-raised)',
              border: `1px solid ${activeNode.name === n.name ? n.border : 'var(--border-subtle)'}`,
              cursor: 'pointer',
              textAlign: 'center',
              transition: 'all var(--transition-med)',
              zIndex: 5,
              minWidth: '95px',
              boxShadow: activeNode.name === n.name ? `0 0 14px ${n.bg}` : 'none',
            }}>
            <div style={{ fontSize: '11px', fontWeight: '700', color: activeNode.name === n.name ? n.color : 'var(--text-secondary)' }}>{n.name}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: activeNode.name === n.name ? n.color : 'var(--text-dim)', fontWeight: '600', marginTop: '2px' }}>
              sens: {n.sensitivity ? n.sensitivity.toFixed(3) : '0.500'}
            </div>
          </button>
        ))}
      </div>

      {/* ─── Node Detail Panel ─── */}
      <div className="panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
          <Info size={14} color="var(--accent-cyan)" /> Stage 3/3b Node Metadata
        </div>

        <div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '3px' }}>Attribute Name</div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', wordBreak: 'break-all' }}>{activeNode.name}</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <div style={{ padding: '10px', background: 'var(--bg-base)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Type</div>
            <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>{isCategorical ? 'Categorical' : 'Numerical'}</div>
          </div>
          <div style={{ padding: '10px', background: 'var(--bg-base)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Sensitivity ($s_j$)</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', fontWeight: '700', color: 'var(--accent-rose)' }}>
              {activeNode.sensitivity ? activeNode.sensitivity.toFixed(3) : '0.500'}
            </div>
          </div>
        </div>

        <div style={{ padding: '10px', background: 'var(--bg-base)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Missingness Rate</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', fontWeight: '700', color: 'var(--accent-amber)' }}>
            {activeNode.missing_pct !== undefined ? `${activeNode.missing_pct}%` : '0.0%'}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>Clinical Graph Rationale</div>
          <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
            Edge strengths are measured on a 300-row sample of the uploaded data for this operator view only. The released model uses the public Stage 3 prior topology W₀ (column names and clinical ontology); in strict mode no private statistic enters the graph.
          </p>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button className="btn btn--primary" style={{ width: '100%', justifyContent: 'center', fontSize: '11px', padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ExternalLink size={12} /> Inspect Graph Adjacency Matrix
          </button>
        </div>
      </div>
    </div>
  );
}
