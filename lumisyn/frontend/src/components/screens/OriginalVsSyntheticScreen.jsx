import React from 'react';
import { BarChart3, CheckCircle2, Info } from 'lucide-react';

function formatMetric(value) {
  return value == null ? '—' : typeof value === 'number' ? value.toFixed(3) : String(value);
}

export default function OriginalVsSyntheticScreen({ comparisonData }) {
  const metrics = comparisonData?.feature_summary || [];
  const classRatio = comparisonData?.class_ratio_summary;
  const hasRun = Boolean(comparisonData?.session_id);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      <div>
        <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <BarChart3 size={16} color="var(--accent-cyan)" /> Measured Original vs Synthetic Comparison
        </h2>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          This page is populated only from the latest backend comparison response; it does not display illustrative patient metrics.
        </p>
      </div>

      {!hasRun ? (
        <div className="panel" style={{ padding: '28px', textAlign: 'center' }}>
          <Info size={24} color="var(--text-muted)" style={{ marginBottom: '8px' }} />
          <div style={{ fontSize: '13px', fontWeight: '700' }}>No completed comparison</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Run the pipeline to populate measured distribution and utility metrics.</div>
        </div>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
            <div className="panel" style={{ padding: '14px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Numeric features compared</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '22px', fontWeight: '700', color: 'var(--accent-cyan)', marginTop: '4px' }}>{metrics.length}</div>
            </div>
            <div className="panel" style={{ padding: '14px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Correlation Frobenius error</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '22px', fontWeight: '700', color: 'var(--accent-violet)', marginTop: '4px' }}>{formatMetric(comparisonData.correlation_frobenius_error)}</div>
            </div>
            <div className="panel" style={{ padding: '14px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Target ratio match</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '22px', fontWeight: '700', color: classRatio?.ratio_matched ? 'var(--accent-emerald)' : 'var(--accent-amber)', marginTop: '4px' }}>{classRatio ? (classRatio.ratio_matched ? 'MATCH' : 'DIFF') : '—'}</div>
            </div>
          </div>

          <div className="panel" style={{ overflow: 'hidden' }}>
            <div className="section-header">
              <span style={{ fontSize: '11px', fontWeight: '700', letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
                Empirical feature summary · {comparisonData.dataset_name}
              </span>
              <span className="badge badge--cyan">MEASURED</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr><th>Feature</th><th>Real mean</th><th>Synthetic mean</th><th>Mean abs. error</th><th>Real std</th><th>Synthetic std</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {metrics.map(metric => (
                    <tr key={metric.feature}>
                      <td className="td-primary">{metric.feature}</td>
                      <td className="td-mono">{formatMetric(metric.real_mean)}</td>
                      <td className="td-mono">{formatMetric(metric.syn_mean)}</td>
                      <td className="td-mono">{formatMetric(metric.mean_abs_error)}</td>
                      <td className="td-mono">{formatMetric(metric.real_std)}</td>
                      <td className="td-mono">{formatMetric(metric.syn_std)}</td>
                      <td><span className="badge badge--emerald"><CheckCircle2 size={9} /> RECORDED</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
