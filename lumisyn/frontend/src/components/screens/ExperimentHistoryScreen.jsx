import React, { useState, useEffect } from 'react';
import { FileSpreadsheet, Search, CheckCircle2, Filter, RefreshCw } from 'lucide-react';
import { apiUrl } from '../../api/client';

const selectStyle = {
  background: 'var(--bg-base)',
  border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-md)',
  padding: '6px 10px',
  fontSize: '11px',
  color: 'var(--text-secondary)',
  fontFamily: 'var(--font-ui)',
  outline: 'none',
  cursor: 'pointer',
};

export default function ExperimentHistoryScreen({ sessionData: _sessionData }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All Status');

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const res = await fetch(apiUrl('/api/runs'));
      if (res.ok) {
        const data = await res.json();
        setRuns(data);
      }
    } catch (err) {
      console.error("Fetch runs error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const displayRuns = runs;

  const filtered = displayRuns.filter(r => {
    const matchStatus = statusFilter === 'All Status' || r.status === statusFilter;
    const matchSearch = !search || r.run_id.toLowerCase().includes(search.toLowerCase()) || r.dataset_name.toLowerCase().includes(search.toLowerCase());
    return matchStatus && matchSearch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">

      {/* ─── Header ─── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <FileSpreadsheet size={16} color="var(--accent-cyan)" />
            Experiment History & Audit Log
          </h2>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Audit log of all ACFM-Net runs, target privacy budgets $\epsilon^*$, DAPC risk scores ($R_P$), and Privacy Decision Certificates.
          </p>
        </div>
        <button onClick={fetchRuns} disabled={loading} className="btn btn--secondary" style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <RefreshCw size={12} className={loading ? "spin" : ""} />
          Refresh Runs
        </button>
      </div>

      {/* ─── Filter Bar ─── */}
      <div className="panel" style={{ padding: '12px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Filter size={12} color="var(--text-muted)" />
          <select style={selectStyle} value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option>All Status</option>
            <option>COMPLETED</option>
            <option>READY</option>
            <option>ABSTAINED</option>
          </select>
        </div>
        <div style={{ position: 'relative' }}>
          <Search size={12} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
          <input
            type="text"
            placeholder="Search runs..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input-field"
            style={{ paddingLeft: '30px', width: '220px', fontSize: '11px' }}
          />
        </div>
      </div>

      {/* ─── Experiments Table ─── */}
      <div className="panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '11px', fontWeight: '700', letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
            Showing {filtered.length} Persistent Experiment Runs
          </span>
          <span className="badge badge--cyan" style={{ fontSize: '10px' }}>DAPC Audited</span>
        </div>

          <table className="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Dataset</th>
              <th>Shape ($N \times D$)</th>
              <th>Selected Target $\epsilon^*$</th>
              <th>Soft-Max Risk $R_P$</th>
              <th>DAPC Decision</th>
              <th>Status</th>
              <th>Created At</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && <tr><td colSpan="8" style={{ padding: '28px', textAlign: 'center', color: 'var(--text-muted)' }}>No persisted experiment runs match the current filter.</td></tr>}
            {filtered.map((run) => (
              <tr key={run.run_id}>
                <td className="td-primary" style={{ fontFamily: 'var(--font-mono)' }}>{run.run_id.slice(0, 8)}</td>
                <td>{run.dataset_name}</td>
                <td className="td-mono">{run.row_count?.toLocaleString()} $\times$ {run.col_count}</td>
                <td className="td-mono" style={{ color: 'var(--accent-emerald)', fontWeight: '700' }}>
                  {run.target_epsilon == null ? '—' : Number(run.target_epsilon).toFixed(3)}
                </td>
                <td className="td-mono" style={{ color: run.risk_score_R_P > 0.5 ? 'var(--accent-amber)' : 'var(--accent-cyan)' }}>
                  {run.risk_score_R_P == null ? '—' : Number(run.risk_score_R_P).toFixed(3)}
                </td>
                <td>
                  <span className={`badge ${run.dapc_decision === 'PROCEED' ? 'badge--emerald' : run.dapc_decision?.includes('ABSTAIN') ? 'badge--rose' : 'badge--cyan'}`}>
                     {run.dapc_decision || '—'}
                  </span>
                </td>
                <td>
                  <span className="badge badge--emerald" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={10} /> {run.status}
                  </span>
                </td>
                <td style={{ color: 'var(--text-dim)', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                   {run.created_at ? new Date(run.created_at * 1000).toLocaleTimeString() : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
