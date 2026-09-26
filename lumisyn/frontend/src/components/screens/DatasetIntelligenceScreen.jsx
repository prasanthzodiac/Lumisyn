import React from 'react';
import { AlertTriangle, CheckCircle2, Layers, Tag, Info } from 'lucide-react';

function EmptyState() {
  return (
    <div className="panel" style={{ padding: '30px', textAlign: 'center' }}>
      <Info size={26} color="var(--text-muted)" style={{ marginBottom: '8px' }} />
      <h3 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)' }}>No dataset loaded</h3>
      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Upload a dataset to inspect its declared schema under the privacy gate.</p>
    </div>
  );
}

export default function DatasetIntelligenceScreen({ sessionData }) {
  if (!sessionData) return <div className="screen-enter"><EmptyState /></div>;

  const columns = sessionData.columns || [];
  const numericCount = sessionData.numeric_columns?.length ?? columns.filter(column => String(column.dtype || '').match(/int|float|double|number/i)).length;
  const categoricalCount = sessionData.categorical_columns?.length ?? Math.max(0, columns.length - numericCount);
  const schemaRows = columns.map(column => typeof column === 'object' ? column : { name: column, dtype: 'unknown' });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[
          { label: 'Rows', value: (sessionData.row_count ?? sessionData.total_rows ?? 0).toLocaleString(), sub: 'Declared dataset size', color: 'var(--accent-cyan)' },
          { label: 'Columns', value: sessionData.column_count ?? sessionData.total_columns ?? columns.length, sub: 'Declared attributes', color: 'var(--text-primary)' },
          { label: 'Numeric', value: numericCount, sub: 'Schema-derived count', color: 'var(--accent-violet)' },
          { label: 'Categorical', value: categoricalCount, sub: 'Schema-derived count', color: 'var(--accent-emerald)' },
        ].map(card => (
          <div className="panel" style={{ padding: '14px' }} key={card.label}>
            <div style={{ fontSize: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>{card.label}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: '700', color: card.color, marginTop: '5px' }}>{card.value}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '3px' }}>{card.sub}</div>
          </div>
        ))}
      </div>

      <div className="panel" style={{ padding: '16px' }}>
        <h3 className="panel-title"><Layers size={14} color="var(--accent-cyan)" /> Dataset Intelligence Boundary</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          <div className="panel" style={{ padding: '12px', background: 'var(--bg-base)' }}><Tag size={14} color="var(--accent-violet)" /><div style={{ fontSize: '11px', marginTop: '5px', fontWeight: '600' }}>Target variable</div><div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '3px' }}>{sessionData.suggested_target || 'Not declared'}</div></div>
          <div className="panel" style={{ padding: '12px', background: 'var(--bg-base)' }}><AlertTriangle size={14} color="var(--accent-amber)" /><div style={{ fontSize: '11px', marginTop: '5px', fontWeight: '600' }}>Missing-value detail</div><div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '3px' }}>{sessionData.missing_cells == null ? 'Withheld by strict privacy gate' : sessionData.missing_cells}</div></div>
          <div className="panel" style={{ padding: '12px', background: 'var(--bg-base)' }}><CheckCircle2 size={14} color="var(--accent-emerald)" /><div style={{ fontSize: '11px', marginTop: '5px', fontWeight: '600' }}>Fingerprint status</div><div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '3px' }}>Generated during pipeline run</div></div>
        </div>
      </div>

      <div className="panel" style={{ overflow: 'hidden' }}>
        <div className="section-header"><span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>Declared schema</span><span className="badge badge--cyan">PRIVACY-SAFE METADATA</span></div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table"><thead><tr><th>Column</th><th>Declared dtype</th><th>Missing detail</th><th>Unique detail</th></tr></thead><tbody>
            {schemaRows.map(row => <tr key={row.name}><td className="td-primary">{row.name}</td><td className="td-mono">{row.dtype || 'unknown'}</td><td>{row.missing_count == null ? 'Withheld' : row.missing_count}</td><td>{row.unique_values == null ? 'Withheld' : row.unique_values}</td></tr>)}
          </tbody></table>
        </div>
      </div>
    </div>
  );
}
