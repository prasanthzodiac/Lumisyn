import React, { useMemo, useState } from 'react';
import { Layers, Activity, Cpu, Info } from 'lucide-react';

const TABS = ['Overview', 'Architecture', 'Training', 'Metrics', 'Outputs'];

export default function StageDetailsHVAEScreen({ generationResult, sessionData }) {
  const [tab, setTab] = useState('Overview');
  const report = useMemo(() => generationResult?.stage_report?.find(item => String(item.stage) === '4'), [generationResult]);
  const preparation = useMemo(() => generationResult?.stage_report?.find(item => String(item.stage) === '2'), [generationResult]);
  const evidence = report?.evidence || {};
  const hasRun = Boolean(report);
  const inputFeatures = preparation?.evidence?.encoded_features;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      <div><h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}><Layers size={16} color="var(--accent-cyan)" /> Stage 4: DP Hierarchical Representation (HCARLN)</h2><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Measured representation evidence from the latest pipeline run for {sessionData?.filename || 'the active dataset'}.</p></div>
      <div className="pill-tabs">{TABS.map(item => <button key={item} onClick={() => setTab(item)} className={`pill-tab ${tab === item ? 'pill-tab--active' : ''}`}>{item}</button>)}</div>

      {!hasRun ? <div className="panel" style={{ padding: '30px', textAlign: 'center' }}><Info size={26} color="var(--text-muted)" style={{ marginBottom: '8px' }} /><h3 style={{ fontSize: '15px' }}>No completed Stage 4 run</h3><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Run synthetic-data generation to populate latent dimensions, steps, and DP-SGD evidence.</p></div> : <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {[['Status', report.status, 'var(--accent-emerald)'], ['Input features', inputFeatures ?? '—', 'var(--accent-cyan)'], ['Latent features', evidence.latent_features ?? '—', 'var(--accent-purple)'], ['DP steps', evidence.steps ?? '—', 'var(--accent-amber)']].map(([label, value, color]) => <div className="panel" style={{ padding: '14px' }} key={label}><div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div><div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: '700', color, marginTop: '4px' }}>{value}</div></div>)}
        </div>
        <div className="panel" style={{ padding: '16px' }}><h3 className="panel-title"><Cpu size={14} color="var(--accent-violet)" /> Measured model components</h3><div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginTop: '12px' }}>{(evidence.trainable_components || []).map(component => <div key={component} className="panel" style={{ padding: '12px', background: 'var(--bg-base)', fontSize: '11px', color: 'var(--text-secondary)' }}>{component}</div>)}</div><div style={{ marginTop: '14px', fontSize: '11px', color: 'var(--text-muted)' }}>Optimizer: {evidence.optimizer || '—'} · Epochs: {evidence.epochs ?? '—'} · View: {tab}</div></div>
        <div className="panel" style={{ padding: '16px' }}><h3 className="panel-title"><Activity size={14} color="var(--accent-cyan)" /> Training evidence</h3><div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '12px' }}><div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Privacy training verified</div><div className="badge badge--emerald" style={{ marginTop: '5px' }}>{generationResult.report_summary?.privacy_training_verified ? 'VERIFIED' : 'CHECK REPORT'}</div></div><div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Accounted epsilon</div><div style={{ fontFamily: 'var(--font-mono)', marginTop: '5px' }}>{generationResult.report_summary?.privacy_composition?.total_epsilon ?? '—'}</div></div><div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Pipeline status</div><div style={{ fontFamily: 'var(--font-mono)', marginTop: '5px' }}>{generationResult.pipeline_status || '—'}</div></div></div></div>
      </>}
    </div>
  );
}
