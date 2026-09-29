import React, { useState } from 'react';
import { Play, Activity, ShieldCheck, Layers, GitBranch, Cpu, Lock, RefreshCw, BarChart2 } from 'lucide-react';
import { apiUrl } from '../api/client';
import SyntheticOutputTransformation from './SyntheticOutputTransformation';

const stageDefinitions = [
  { id: '0', title: 'Stage 0: Governance (CDGL)', desc: 'Safe Harbor de-identification and audit handle', icon: ShieldCheck },
  { id: '1', title: 'Stage 1: Intelligence (UDIE)', desc: 'Measured dataset task and structure analysis', icon: Cpu },
  { id: '2', title: 'Stage 2: Preparation (ADPE)', desc: 'Privacy-safe imputation, encoding and scaling', icon: Layers },
  { id: '3', title: 'Stage 3: Domain Constraint Graph', desc: 'DP-sanitized dependency graph construction', icon: GitBranch },
  { id: '5', title: 'Stage 5: Privacy Controller (APC)', desc: 'Adaptive target and release ledger composition', icon: Lock },
  { id: '4', title: 'Stage 4: Representation (HCARLN)', desc: 'DP-SGD graph-attention and hierarchical VAE', icon: Activity },
  { id: '3b', title: 'Stage 3b: Post-Latent Refinement', desc: 'Latent-informed graph topology refinement', icon: RefreshCw },
  { id: '6', title: 'Stage 6: Latent Generator (OT-CFM)', desc: 'Continuous latent flow field initialization', icon: Play },
  { id: '8', title: 'Stage 8: DP Optimization', desc: 'Measured DP-SGD flow training diagnostics', icon: BarChart2 },
  { id: '7', title: 'Stage 7: Constrained Repair Loop', desc: 'Measured domain-constraint repair statistics', icon: RefreshCw },
  { id: '7c', title: 'Stage 7c: DP Class-Conditional Marginals', desc: 'Noisy label × feature tables, jointly accounted', icon: Lock },
  { id: '9', title: 'Stage 9: Synthetic Data Output', desc: 'Actual decoded rows through repair and release', icon: Layers },
];

const statusColors = {
  PASSED: { text: '#065f46', bg: 'rgba(16,185,129,0.06)', border: 'rgba(16,185,129,0.15)' },
  READY: { text: '#065f46', bg: 'rgba(16,185,129,0.06)', border: 'rgba(16,185,129,0.15)' },
  INITIALIZED: { text: '#92400e', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)' },
  UNVERIFIED: { text: '#991b1b', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.2)' },
  RUNNING: { text: '#3730a3', bg: 'rgba(99,102,241,0.08)', border: 'rgba(99,102,241,0.2)' },
};

export default function PipelineVisualizer({ sessionData, onGenerationComplete, initialResult = null }) {
  const [samples, setSamples] = useState(1000);
  const [epsilon, setEpsilon] = useState(1.0);
  const [epochs, setEpochs] = useState(10);
  const [generating, setGenerating] = useState(false);
  const [activeStage, setActiveStage] = useState(0);
  const [result, setResult] = useState(initialResult);

  const stages = stageDefinitions;
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!sessionData || !sessionData.session_id) {
      alert("Please upload or select a dataset first on the Landing Page!");
      return;
    }

    setGenerating(true);
    setError(null);
    setResult(null);
    setActiveStage(0);

    try {
      const generationRequest = fetch(apiUrl('/api/generate'), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionData.session_id,
          num_samples: parseInt(samples),
          target_epsilon: parseFloat(epsilon),
          epochs: parseInt(epochs),
          target_col: sessionData.suggested_target
        })
      });
      // Cycle the stage highlight while the real backend run executes; statuses come from its measured records.
      let done = false;
      generationRequest.finally(() => { done = true; }).catch(() => {});
      for (let i = 0; !done; i++) {
        setActiveStage(i % stages.length);
        await new Promise((r) => setTimeout(r, 400));
      }
      const res = await generationRequest;
      if (!res.ok) {
        let detail = `Generation failed (${res.status})`;
        try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error body */ }
        throw new Error(detail);
      }
      const data = await res.json();
      setResult(data);
      if (onGenerationComplete) onGenerationComplete(data);
    } catch (err) {
      console.error("Generation error:", err);
      setError(err.message || 'Generation failed.');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h3 className="panel-title" style={{ marginBottom: '4px' }}>
          <Activity size={15} color="var(--accent-violet)" />
          Lumisyn-Net v5 Real-Time Synthesizer (Stages 0-9)
        </h3>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Configure generation hyper-parameters and trigger the measured 11-record execution sequence in the PyTorch backend.
        </p>
      </div>

      {/* Parameter Control Panel */}
      <div className="panel" style={{ padding: '16px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '9px', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>Synthetic Samples</label>
          <input
            type="number"
            value={samples}
            onChange={(e) => setSamples(e.target.value)}
            className="input-field"
            style={{ fontFamily: 'var(--font-mono)' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '9px', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>DP Budget Target (ε*)</label>
          <input
            type="number"
            step="0.1"
            value={epsilon}
            onChange={(e) => setEpsilon(e.target.value)}
            className="input-field"
            style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '9px', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>Training Epochs</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            className="input-field"
            style={{ fontFamily: 'var(--font-mono)' }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', height: '36px' }}
          >
            <Play size={13} className={generating ? 'animate-spin' : ''} />
            <span>{generating ? 'Running pipeline...' : 'Execute Lumisyn-Net'}</span>
          </button>
        </div>
      </div>

      {/* Measured stage visualizer */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
        {stages.map((stg, idx) => {
          const Icon = stg.icon;
          const measured = result?.stage_report?.find(item => String(item.stage) === stg.id);
          const isActive = generating && idx === activeStage;
          const status = measured?.status || (isActive ? 'RUNNING' : generating ? 'PENDING' : 'WAITING');
          const statusStyle = statusColors[status] || { text: 'var(--text-muted)', bg: 'rgba(0,0,0,0.03)', border: 'transparent' };
          const isComplete = status === 'PASSED' || status === 'READY';

          return (
            <div
              key={stg.id}
              className="panel"
              style={{
                padding: '12px',
                background: isActive ? 'rgba(99,102,241,0.04)' : 'var(--bg-panel)',
                borderColor: isActive ? 'var(--accent-violet)' : isComplete ? 'rgba(16,185,129,0.3)' : status === 'UNVERIFIED' ? 'rgba(239,68,68,0.3)' : 'var(--border-default)',
                transform: isActive ? 'scale(1.02)' : 'none',
                transition: 'border-color var(--transition-med), box-shadow var(--transition-med), transform var(--transition-med)',
                boxShadow: isActive ? 'var(--glow-violet)' : 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <Icon size={14} style={{ color: isComplete ? 'var(--accent-emerald)' : isActive ? 'var(--accent-violet)' : 'var(--text-muted)' }} />
                <span className="badge" style={{
                  background: statusStyle.bg,
                  color: statusStyle.text,
                  borderColor: statusStyle.border,
                }}>
                  {status}
                </span>
              </div>
              <h4 style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '3px' }}>{stg.title}</h4>
              <p style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.4 }}>{stg.desc}</p>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="panel" style={{ padding: '12px 14px', borderColor: 'rgba(239,68,68,0.3)', color: 'var(--accent-rose)', fontSize: '11px' }}>
          {error}
        </div>
      )}

      <SyntheticOutputTransformation
        preview={result?.synthetic_preview || []}
        columns={(result?.synthetic_preview?.[0] && Object.keys(result.synthetic_preview[0])) || []}
      />

      {/* Execution Results Summary Banner */}
      {result && (
        <div className="panel" style={{
          padding: '16px',
          borderColor: 'rgba(16,185,129,0.3)',
          background: 'rgba(16,185,129,0.03)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ShieldCheck size={16} color="var(--accent-emerald)" /> Pipeline Execution: {result.pipeline_status || 'COMPLETED'}
            </h3>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Generated {result.generated_samples} synthetic samples in {result.generation_time_sec}s; {result.stages_passed ?? 0} measured stages passed.
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>PRV Accountant Epsilon Consumed</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: '800', color: 'var(--accent-emerald)' }}>ε = {result.report_summary?.prv_eps_used}</div>
          </div>
        </div>
      )}
    </div>
  );
}
