import React from 'react';
import { FileDown, FileText } from 'lucide-react';
import { apiUrl } from '../api/client';

const STAGE_NAMES = {
  '0': 'Governance', '1': 'Intelligence', '2': 'Preparation', '3': 'Constraint graph', '3b': 'Graph refinement',
  '5': 'Privacy controller', '4': 'Representation', '6': 'Flow matching', '8': 'DP flow training',
  '7': 'Constrained repair', '7c': 'DP marginals', '7b': 'DP recalibration', '9': 'Synthetic output',
};

const tone = (status) =>
  status === 'PASSED' || status === 'READY' ? 'badge--emerald' : status === 'SKIPPED' ? '' : 'badge--rose';

// Per-stage PDF reports of the latest run, plus one combined PDF.
export default function StageReportLinks({ sessionId, stageReport, title = 'Stage Reports (PDF)' }) {
  if (!sessionId || !stageReport?.length) return null;
  return (
    <div className="panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="panel-title" style={{ margin: 0 }}>
          <FileText size={15} color="var(--accent-violet)" /> {title}
        </h3>
        <a className="btn-primary" href={apiUrl(`/api/report/stages/${sessionId}`)} style={{ textDecoration: 'none', height: '32px' }}>
          <FileDown size={13} /> <span>All stages (PDF)</span>
        </a>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '8px' }}>
        {stageReport.map((s) => (
          <a
            key={s.stage}
            href={apiUrl(`/api/report/stage/${sessionId}/${s.stage}`)}
            title={`Download the Stage ${s.stage} report`}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
              padding: '8px 10px', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
              textDecoration: 'none', color: 'var(--text-primary)', background: 'var(--bg-panel)', fontSize: '11px',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
              <FileDown size={12} color="var(--accent-violet)" />
              <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                <b>{s.stage}</b> · {STAGE_NAMES[s.stage] || s.name}
              </span>
            </span>
            <span className={`badge ${tone(s.status)}`} style={{ fontSize: '9px', flexShrink: 0 }}>{s.status}</span>
          </a>
        ))}
      </div>
    </div>
  );
}
