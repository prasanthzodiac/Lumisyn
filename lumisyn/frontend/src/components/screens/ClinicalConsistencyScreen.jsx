import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

export default function ClinicalConsistencyScreen({ comparisonData }) {
  const hasRun = Boolean(comparisonData?.session_id);
  const violations = comparisonData?.clinical_rule_violations;
  const passRate = comparisonData?.clinical_rule_pass_rate;
  const passed = hasRun && Number(violations) === 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      <div>
        <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <ShieldCheck size={16} color="var(--accent-emerald)" /> Measured Clinical Consistency
        </h2>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Reports the constraint checks returned by the latest synthetic-data comparison. No fallback clinical score is displayed.
        </p>
      </div>

      {!hasRun ? (
        <div className="panel" style={{ padding: '28px', textAlign: 'center' }}>
          <Info size={24} color="var(--text-muted)" style={{ marginBottom: '8px' }} />
          <div style={{ fontSize: '13px', fontWeight: '700' }}>No clinical consistency result</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Run the pipeline and comparison audit first.</div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          <div className="panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Rule violations</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', fontWeight: '700', color: passed ? 'var(--accent-emerald)' : 'var(--accent-rose)', marginTop: '4px' }}>{violations ?? '—'}</div>
          </div>
          <div className="panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Measured pass rate</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', fontWeight: '700', color: 'var(--accent-cyan)', marginTop: '4px' }}>{passRate == null ? '—' : `${passRate}%`}</div>
          </div>
          <div className="panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Status</div>
            <div style={{ marginTop: '8px' }}><span className={`badge ${passed ? 'badge--emerald' : 'badge--amber'}`}>{passed ? 'MEASURED PASS' : 'REVIEW REQUIRED'}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
