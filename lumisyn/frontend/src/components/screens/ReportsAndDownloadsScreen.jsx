import React, { useEffect, useState } from 'react';
import { FileText, Download, FileCheck, Zap, Lock } from 'lucide-react';
import { apiUrl } from '../../api/client';
import StageReportLinks from '../StageReportLinks';

const reportCards = [
  { title: "PDF Report",   desc: "Full academic & clinical validation report PDF",   color: 'rose',   format: 'PDF' },
  { title: "Privacy Certificate", desc: "Immutable Privacy Decision Certificate (PDC) JSON", color: 'emerald', format: 'PDC' },
];

const artifactCards = [
  { title: "Synthetic Dataset",   desc: "Synthetic patient records CSV",         icon: FileText,  format: 'CSV',  color: 'cyan'    },
  { title: "Power BI Report",     desc: "Pre-configured PBIX data model package", icon: FileCheck, format: 'PBIX', color: 'violet'  },
  { title: "Privacy Certificate", desc: "Signed Privacy Decision Certificate",    icon: Lock,      format: 'PDC',  color: 'emerald' },
];

const colorClasses = {
  rose:    { badge: 'badge--rose',    icon: 'icon-badge--rose',    btn: 'rgba(244,63,94,0.15)',    border: 'rgba(244,63,94,0.3)',    text: 'var(--accent-rose)' },
  cyan:    { badge: 'badge--cyan',    icon: 'icon-badge--cyan',    btn: 'rgba(0,212,255,0.1)',     border: 'rgba(0,212,255,0.25)',   text: 'var(--accent-cyan)' },
  violet:  { badge: 'badge--violet',  icon: 'icon-badge--violet',  btn: 'rgba(168,85,247,0.12)',   border: 'rgba(168,85,247,0.3)',   text: 'var(--accent-purple)' },
  emerald: { badge: 'badge--emerald', icon: 'icon-badge--emerald', btn: 'rgba(16,185,129,0.1)',    border: 'rgba(16,185,129,0.25)',  text: 'var(--accent-emerald)' },
  amber:   { badge: 'badge--amber',   icon: 'icon-badge--amber',   btn: 'rgba(245,158,11,0.1)',    border: 'rgba(245,158,11,0.25)', text: 'var(--accent-amber)' },
};

export default function ReportsAndDownloadsScreen({ sessionData }) {
  const [stageReport, setStageReport] = useState([]);
  useEffect(() => {
    if (!sessionData?.session_id) return undefined;
    let active = true;
    fetch(apiUrl(`/api/stages/${sessionData.session_id}`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (active) setStageReport(d?.stage_reports || []); })
      .catch(() => {});
    return () => { active = false; };
  }, [sessionData?.session_id, sessionData?.lastRunAt]);

  const handleDownload = (format) => {
    if (!sessionData?.session_id) {
      alert("No active session found. Please upload a dataset first.");
      return;
    }
    if (format === 'PDF') {
      window.open(apiUrl(`/api/report/pdf/${sessionData.session_id}`));
    } else if (format === 'CSV') {
      window.open(apiUrl(`/api/download/${sessionData.session_id}`));
    } else if (format === 'PBIX') {
      window.open(apiUrl(`/api/powerbi/export/${sessionData.session_id}`));
    } else if (format === 'PDC') {
      window.open(apiUrl(`/api/dapc/certificate/${sessionData.session_id}`));
    } else {
      alert(`Exporting ${format} package for active dataset: ${sessionData.filename}`);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }} className="screen-enter">

      {/* ─── Header ─── */}
      <div>
        <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <FileText size={16} color="var(--accent-cyan)" />
          Stage 14: Reports Generator & Artifact Downloads
        </h2>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Export publication-ready research reports and download synthetic data packages.
        </p>
      </div>

      {/* ─── Generate Reports ─── */}
      <div>
        <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Zap size={11} color="var(--accent-cyan)" />
          Generate Academic & Clinical Reports
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {reportCards.map((rc, i) => {
            const c = colorClasses[rc.color];
            return (
              <div key={rc.title} className="panel animate-in" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', animationDelay: `${i * 60}ms` }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div className={`icon-badge ${c.icon}`}>
                    <FileText size={15} />
                  </div>
                  <span className={`badge ${c.badge}`} style={{ fontFamily: 'var(--font-mono)', fontSize: '9px' }}>{rc.format}</span>
                </div>
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px' }}>{rc.title}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{rc.desc}</div>
                </div>
                <button
                  className="btn-primary"
                  onClick={() => handleDownload(rc.format)}
                  disabled={!sessionData?.session_id}
                  style={{
                    width: '100%',
                    justifyContent: 'center',
                    background: c.btn,
                    borderColor: c.border,
                    color: c.text,
                    marginTop: 'auto',
                    cursor: sessionData?.session_id ? 'pointer' : 'not-allowed',
                    opacity: sessionData?.session_id ? 1 : 0.5
                  }}
                >
                  <Zap size={11} />
                  Generate & Download
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* ─── Download Artifacts ─── */}
      <div>
        <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Download size={11} color="var(--accent-cyan)" />
          Download Experiment Artifacts
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {artifactCards.map((ac, i) => {
            const Icon = ac.icon;
            const c = colorClasses[ac.color];
            return (
              <div key={ac.title} className="panel animate-in" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', animationDelay: `${260 + i * 60}ms` }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div className={`icon-badge ${c.icon}`}>
                    <Icon size={15} />
                  </div>
                  <span className={`badge ${c.badge}`} style={{ fontFamily: 'var(--font-mono)', fontSize: '9px' }}>{ac.format}</span>
                </div>
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px' }}>{ac.title}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    {ac.title === "Synthetic Dataset" && sessionData 
                      ? `${sessionData.row_count?.toLocaleString()} synthetic patient records CSV` 
                      : ac.desc}
                  </div>
                </div>
                <button
                  onClick={() => handleDownload(ac.format)}
                  disabled={!sessionData?.session_id}
                  style={{
                    width: '100%',
                    padding: '7px 12px',
                    borderRadius: 'var(--radius-md)',
                    background: 'transparent',
                    border: '1px solid var(--border-default)',
                    color: 'var(--text-secondary)',
                    fontSize: '11px',
                    fontWeight: '600',
                    cursor: sessionData?.session_id ? 'pointer' : 'not-allowed',
                    opacity: sessionData?.session_id ? 1 : 0.5,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px',
                    transition: 'all var(--transition-fast)',
                    fontFamily: 'var(--font-ui)',
                    marginTop: 'auto',
                  }}
                >
                  <Download size={12} />
                  Download
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <StageReportLinks sessionId={sessionData?.session_id} stageReport={stageReport} title="Per-Stage Reports (PDF)" />
    </div>
  );
}
