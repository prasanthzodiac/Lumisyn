import React, { useCallback, useState, useEffect } from 'react';
import { Lock, ShieldCheck, AlertTriangle, FileText, CheckCircle2, XCircle, Info, RefreshCw, Layers } from 'lucide-react';
import { apiUrl } from '../../api/client';

export default function AdaptivePrivacyControllerScreen({ sessionData, comparisonData }) {
  const [loading, setLoading] = useState(false);
  const [dapcCert, setDapcCert] = useState(comparisonData?.dapc_certificate || null);
  const [error, setError] = useState(null);

  const fetchDapcAnalysis = useCallback(async () => {
    if (!sessionData?.session_id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl(`/api/dapc/analyze/${sessionData.session_id}`));
      if (!res.ok) throw new Error("Failed to analyze dataset privacy risk.");
      const cert = await res.json();
      setDapcCert(cert);
    } catch (err) {
      console.error("DAPC Fetch Error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [sessionData?.session_id]);

  useEffect(() => {
    if (sessionData?.session_id && !dapcCert) {
      fetchDapcAnalysis();
    }
  }, [sessionData?.session_id, dapcCert, fetchDapcAnalysis]);

  const epsSelected = dapcCert?.selected_target_epsilon || comparisonData?.privacy?.target_epsilon || null;
  const epsUsed = comparisonData?.privacy?.eps_used || 0.0;
  const riskScore = dapcCert?.risk_score_R_P ?? null;
  const decision = dapcCert?.decision || "AWAITING_ANALYSIS";
  const riskBreakdown = dapcCert?.risk_breakdown || [];

  if (!sessionData?.session_id) {
    return <div className="screen-enter"><div className="panel" style={{ padding: '30px', textAlign: 'center' }}><Info size={26} color="var(--text-muted)" style={{ marginBottom: '8px' }} /><h3 style={{ fontSize: '15px' }}>No dataset selected</h3><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Upload a dataset before analyzing the adaptive privacy budget.</p></div></div>;
  }
  if (epsSelected == null || riskScore == null) {
    return <div className="screen-enter"><div className="panel" style={{ padding: '30px', textAlign: 'center' }}><RefreshCw size={26} color="var(--accent-cyan)" className={loading ? 'spin' : ''} style={{ marginBottom: '8px' }} /><h3 style={{ fontSize: '15px' }}>{loading ? 'Analyzing privacy risk…' : 'Privacy analysis unavailable'}</h3><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{error || 'Run the privacy analysis again after the dataset is loaded.'}</p></div></div>;
  }

  // SVG Gauge Donut Calculation
  const maxEps = 1.0;
  const pct = Math.min(epsSelected / maxEps, 1);
  const circumference = 2 * Math.PI * 44;
  const dashArray = `${pct * circumference} ${circumference}`;

  const getDecisionBadge = (dec) => {
    switch (dec) {
      case 'PROCEED':
        return <span className="badge badge--amber" style={{ padding: '6px 14px', fontSize: '11px', display: 'inline-flex', alignItems: 'center', gap: '5px' }}><CheckCircle2 size={12} /> PROCEED — POLICY REVIEW</span>;
      case 'PROCEED_STRICT_PRIVACY':
        return <span className="badge badge--cyan" style={{ padding: '6px 14px', fontSize: '11px', display: 'inline-flex', alignItems: 'center', gap: '5px' }}><ShieldCheck size={12} /> PROCEED WITH STRICT PRIVACY</span>;
      case 'ABSTAIN':
        return <span className="badge badge--rose" style={{ padding: '6px 14px', fontSize: '11px', display: 'inline-flex', alignItems: 'center', gap: '5px' }}><AlertTriangle size={12} /> SECURITY ABSTAIN — RE-ID VULNERABILITY</span>;
      case 'REJECT':
        return <span className="badge badge--rose" style={{ padding: '6px 14px', fontSize: '11px', display: 'inline-flex', alignItems: 'center', gap: '5px' }}><XCircle size={12} /> HIGH RISK REJECT</span>;
      default:
        return <span className="badge badge--amber" style={{ padding: '6px 14px', fontSize: '11px' }}>EVALUATING</span>;
    }
  };

  const handleDownloadCertificate = () => {
    if (!dapcCert) return;
    const jsonStr = JSON.stringify(dapcCert, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Privacy_Decision_Certificate_${dapcCert.certificate_id || 'PDC'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      {error && <div className="badge badge--rose">{error}</div>}

      {/* ─── Top Bar & Header ─── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-card)', padding: '16px 20px', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-default)' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Lock size={18} color="var(--accent-emerald)" />
            Stage 5: Dataset-Adaptive Privacy Budget Controller (DAPC v6)
          </h2>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            DP-Sanitized Meta-Selection Engine with Soft-Max Monotonic Risk Model (R_P) & Policy Envelope [0.40, 1.00].
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {getDecisionBadge(decision)}
          <button
            onClick={fetchDapcAnalysis}
            disabled={loading}
            className="btn btn--secondary"
            style={{ fontSize: '11px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <RefreshCw size={12} className={loading ? "spin" : ""} />
            Re-Analyze Risk
          </button>
        </div>
      </div>

      {/* ─── Security Warning Banner (if Abstaining) ─── */}
      {(decision === 'ABSTAIN' || decision === 'REJECT') && (
        <div style={{ padding: '14px 18px', background: 'rgba(244,63,94,0.10)', border: '1px solid rgba(244,63,94,0.30)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
          <AlertTriangle size={18} color="var(--accent-rose)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <h4 style={{ fontSize: '13px', fontWeight: '700', color: 'var(--accent-rose)', marginBottom: '4px' }}>
              DAPC Defensive Abstention Triggered — Training Blocked
            </h4>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {dapcCert?.decision_reasons?.[0] || "Critical sample uniqueness combined with small cohort size poses elevated re-identification risk."}
              <br />
              <strong>Required Action:</strong> Conduct manual institutional governance review before overriding privacy parameters.
            </p>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.3fr', gap: '16px' }}>

        {/* ─── Left Panel: Epsilon Gauge & Policy Envelope ─── */}
        <div className="panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '18px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: '700', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Adaptive Privacy Target — ε*
          </div>

          {/* SVG Donut Gauge */}
          <div style={{ position: 'relative', width: '150px', height: '150px' }}>
            <svg width="150" height="150" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="75" cy="75" r="44" fill="none" stroke="var(--border-default)" strokeWidth="10" />
              <circle
                cx="75" cy="75" r="44"
                fill="none"
                stroke="url(#dapcGrad)"
                strokeWidth="10"
                strokeDasharray={dashArray}
                strokeLinecap="round"
                style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(0.16,1,0.3,1)' }}
              />
              <defs>
                <linearGradient id="dapcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="var(--accent-cyan)" />
                  <stop offset="100%" stopColor="var(--accent-emerald)" />
                </linearGradient>
              </defs>
            </svg>
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexDirection: 'column',
            }}>
              <span style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: '600', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Selected ε*</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '32px', fontWeight: '700', color: 'var(--accent-emerald)', lineHeight: 1 }}>
                {epsSelected.toFixed(3)}
              </span>
              <span className="badge badge--emerald" style={{ marginTop: '4px', fontSize: '9px' }}>
                δ = 10⁻⁵
              </span>
            </div>
          </div>

          {/* Policy Envelope Bounds & Meta Partition */}
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '8px', background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <span>Policy Envelope Range:</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--text-primary)' }}>
                [ε_min = 0.40, ε_max = 1.00]
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <span>DP Meta-Budget (ε_meta):</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-cyan)' }}>
                0.020 (Sanitized Selection)
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <span>Training Budget (ε_SGD):</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-purple)' }}>
                {(epsSelected - 0.02).toFixed(3)}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
              <span>Calibrated Noise (σ*):</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-amber)' }}>
                {dapcCert?.calibrated_noise_multiplier_sigma || 1.48}
              </span>
            </div>
          </div>

          {/* Dual Accountant Gauge */}
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: '600', color: 'var(--text-secondary)' }}>
              <span>PRV Accountant Spending</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                {epsUsed.toFixed(4)} / {epsSelected.toFixed(3)}
              </span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'var(--bg-base)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, (epsUsed / epsSelected) * 100)}%`, height: '100%', background: 'var(--accent-emerald)', transition: 'width 0.5s' }} />
            </div>
          </div>

          <button
            onClick={handleDownloadCertificate}
            className="btn btn--primary"
            style={{ width: '100%', fontSize: '11px', padding: '9px 14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
          >
            <FileText size={14} />
            Download Privacy Decision Certificate (PDC)
          </button>
        </div>

        {/* ─── Right Panel: 7D Risk Vector & Soft-Max R_P ─── */}
        <div className="panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 className="panel-title" style={{ margin: 0 }}>
              <Layers size={14} color="var(--accent-cyan)" />
              Dataset Privacy Risk Vector (r_data ∈ [0, 1]⁷)
            </h3>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Soft-Max Risk R_P: </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: '700', color: riskScore > 0.6 ? 'var(--accent-rose)' : 'var(--accent-cyan)' }}>
                {riskScore.toFixed(3)}
              </span>
            </div>
          </div>

          <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: 0 }}>
            Soft-Max Monotonic Risk Aggregator: R_P = 1 - ∏_{"{k=1}"}⁷ (1 - w_k r_k)^{"{γ_k}"}. Dominates high vulnerability dimensions.
          </p>

          {/* 7D Risk Breakdown Bars */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {riskBreakdown.map((item, idx) => (
              <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--text-secondary)', fontWeight: '600' }}>{item.name}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: item.score > 0.6 ? 'var(--accent-rose)' : item.score > 0.3 ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                    {item.score.toFixed(3)} (w={item.weight})
                  </span>
                </div>
                <div style={{ width: '100%', height: '7px', background: 'var(--bg-base)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, item.score * 100)}%`,
                      height: '100%',
                      background: item.score > 0.6 ? 'var(--accent-rose)' : item.score > 0.3 ? 'var(--accent-amber)' : 'var(--accent-emerald)',
                      transition: 'width 0.4s ease'
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Mathematical Non-Linear Response Panel */}
          <div style={{ padding: '12px 14px', background: 'var(--bg-base)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
            <Info size={16} color="var(--accent-violet)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '3px' }}>
                Policy-Bounded Non-Linear Budget Mapping (β = 2.5)
              </div>
              <p style={{ fontSize: '10.5px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                High risk-aversion concavity maps risk R_P = {riskScore.toFixed(3)} strictly into policy envelope [0.40, 1.00], yielding optimal ε* = {epsSelected.toFixed(3)}. Dynamic gradient norm weights v_j = exp(λ s_j) scale coordinate sensitivity inside global vector DP-SGD.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
