import React, { useEffect, useState } from 'react';
import { Database, Cpu, GitBranch, Layers, Lock, Play, ShieldCheck, BarChart2, FileSpreadsheet, CheckCircle2, ChevronRight } from 'lucide-react';
import PipelineVisualizer from '../PipelineVisualizer';
import { apiUrl } from '../../api/client';

const pipelineStages = [
  { id: '0',  name: "Governance",       sub: "De-identification", icon: Database,       color: 'cyan'   },
  { id: '1',  name: "Analyzer",         sub: "UDIE Engine",       icon: Cpu,            color: 'cyan'   },
  { id: '2',  name: "Preparation",      sub: "ADPE",               icon: FileSpreadsheet,color: 'cyan'   },
  { id: '3',  name: "Knowledge Graph",  sub: "Clinical KG",        icon: GitBranch,      color: 'violet' },
  { id: '5',  name: "Privacy (APC)",    sub: "DP Budget",           icon: Lock,          color: 'emerald'},
  { id: '4',  name: "HVAE",             sub: "DP Representation",  icon: Layers,        color: 'violet' },
  { id: '3b', name: "Graph Refinement", sub: "Post-latent",         icon: GitBranch,      color: 'violet' },
  { id: '6',  name: "Flow Matching",    sub: "DP Generator",        icon: Play,          color: 'cyan'   },
  { id: '8',  name: "Optimization",     sub: "DP Training",          icon: BarChart2,     color: 'violet' },
  { id: '7',  name: "Repair",           sub: "Constraint pass",     icon: ShieldCheck,   color: 'cyan'   },
  { id: '9',  name: "Synthetic Data",   sub: "Measured output",     icon: FileSpreadsheet,color: 'emerald'},
];

const colorMap = {
  cyan:    { icon: 'var(--accent-cyan)',    bg: 'rgba(6, 182, 212, 0.08)',   border: 'rgba(6, 182, 212, 0.18)' },
  violet:  { icon: 'var(--accent-violet)',  bg: 'rgba(99, 102, 241, 0.08)',   border: 'rgba(99, 102, 241, 0.18)' },
  emerald: { icon: 'var(--accent-emerald)', bg: 'rgba(16, 185, 129, 0.08)',   border: 'rgba(16, 185, 129, 0.18)' },
};

export default function PipelineOverviewScreen({ sessionData, onGenerationComplete, generationResult }) {
  const [stageReport, setStageReport] = useState(null);

  useEffect(() => {
    if (!sessionData?.session_id) {
      setStageReport(null);
      return undefined;
    }
    let active = true;
    fetch(apiUrl(`/api/stages/${sessionData.session_id}`))
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (active) setStageReport(data); })
      .catch(() => { if (active) setStageReport(null); });
    return () => { active = false; };
  }, [sessionData?.session_id, sessionData?.lastRunAt, generationResult?.generation_time_sec]);

  const reports = stageReport?.stage_reports || [];
  const reportFor = stage => reports.find(item => String(item.stage) === String(stage));
  const evidenceText = stage => {
    const evidence = reportFor(stage)?.evidence || {};
    return Object.entries(evidence).slice(0, 3).map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`).join(' · ');
  };
  const runtime = stageReport?.generation_runtime_seconds;
  const status = stageReport?.pipeline_status || 'READY';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }} className="screen-enter">

      {/* ─── Real-Time Synthesizer ─── */}
      <div className="panel animate-in" style={{ padding: '20px' }}>
        <PipelineVisualizer sessionData={sessionData} onGenerationComplete={onGenerationComplete} />
      </div>

      {/* ─── Pipeline Flow ─── */}
      <div className="panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h3 className="panel-title" style={{ marginBottom: 0 }}>
            <Layers size={14} color="var(--accent-cyan)" />
            Lumisyn-Net v5 — End-to-End Pipeline Workflow
          </h3>
          <span className={`badge ${status === 'SUCCESS' ? 'badge--emerald' : 'badge--cyan'}`}>
            <CheckCircle2 size={9} /> {status}
          </span>
        </div>

        {/* Stage Grid */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0', overflowX: 'auto', padding: '8px 0' }}>
          {pipelineStages.map((stg, i) => {
            const Icon = stg.icon;
            const c = colorMap[stg.color];
            const measured = reportFor(stg.id);
            const isPassed = measured?.status === 'PASSED' || measured?.status === 'READY';
            return (
              <React.Fragment key={stg.id}>
                <div className="pipeline-node animate-in" style={{ animationDelay: `${i * 50}ms` }}>
                  <div style={{
                    width: '52px', height: '52px',
                    borderRadius: 'var(--radius-lg)',
                    background: c.bg,
                    border: `1px solid ${c.border}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: c.icon,
                    transition: 'all var(--transition-med)',
                    cursor: 'default',
                  }}
                    onMouseEnter={e => {
                      e.currentTarget.style.boxShadow = `0 0 16px ${c.border}`;
                      e.currentTarget.style.borderColor = c.icon.replace('var(', '').replace(')', '');
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.boxShadow = 'none';
                      e.currentTarget.style.borderColor = c.border;
                    }}
                  >
                    <Icon size={20} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '1px' }}>
                      {stg.name}
                    </div>
                    <div style={{ fontSize: '9px', color: 'var(--text-dim)' }}>{stg.sub}</div>
                  </div>
                  <div style={{
                    width: '16px', height: '16px',
                    borderRadius: '50%',
                    background: isPassed ? 'rgba(16,185,129,0.15)' : 'rgba(148,163,184,0.12)',
                    border: `1px solid ${isPassed ? 'rgba(16,185,129,0.3)' : 'rgba(148,163,184,0.25)'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <CheckCircle2 size={9} color={isPassed ? 'var(--accent-emerald)' : 'var(--text-dim)'} />
                  </div>
                </div>

                {i < pipelineStages.length - 1 && (
                  <div style={{ display: 'flex', alignItems: 'center', padding: '0 6px', paddingBottom: '32px', flexShrink: 0 }}>
                    <div style={{
                      width: '28px', height: '1px',
                      background: `linear-gradient(90deg, ${colorMap[stg.color].border}, ${colorMap[pipelineStages[i+1].color].border})`,
                    }} />
                    <ChevronRight size={10} style={{ color: 'var(--text-dim)', marginLeft: '-4px' }} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* ─── Stage Progress Details ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
        {pipelineStages.map((stg, i) => {
          const c = colorMap[stg.color];
          const Icon = stg.icon;
          const measured = reportFor(stg.id);
          const progress = measured ? (measured.status === 'PASSED' || measured.status === 'READY' ? 100 : 50) : 0;
          return (
            <div key={stg.id} className="panel animate-in" style={{ padding: '14px', animationDelay: `${i * 40}ms` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                <div style={{
                  width: '28px', height: '28px',
                  borderRadius: '7px',
                  background: c.bg,
                  border: `1px solid ${c.border}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: c.icon,
                  flexShrink: 0,
                }}>
                  <Icon size={13} />
                </div>
                <div>
                  <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-primary)' }}>{stg.name}</div>
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{stg.sub}</div>
                </div>
                <span className={`badge ${progress === 100 ? 'badge--emerald' : 'badge--cyan'}`} style={{ marginLeft: 'auto' }}>{measured?.status || 'Pending'}</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill progress-fill--emerald" style={{ width: `${progress}%` }} />
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginTop: '5px', fontFamily: 'var(--font-mono)' }}>
                {progress}% · Stage {i + 1} of {pipelineStages.length}
              </div>
              {measured && <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '6px', lineHeight: 1.4 }}>{evidenceText(stg.id)}</div>}
            </div>
          );
        })}
      </div>

      {/* ─── Status Footer ─── */}
      <div className="panel" style={{
        padding: '16px 20px',
        border: '1px solid rgba(16,185,129,0.2)',
        background: 'rgba(16,185,129,0.04)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="live-dot" />
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Pipeline Status</div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-primary)' }}>{status === 'READY' ? 'No generation run yet' : `${reports.length} measured stages reported`}</div>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Total Runtime</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: '700', color: 'var(--accent-emerald)' }}>{runtime == null ? '—' : `${runtime.toFixed(2)}s`}</div>
        </div>
      </div>
    </div>
  );
}
