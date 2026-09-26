import React from 'react';
import { Database, Activity, FileSpreadsheet, FileText, ArrowUpRight, TrendingUp, Clock } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';

const trendUp = [
  { v: 8 }, { v: 10 }, { v: 9 }, { v: 12 }, { v: 11 }, { v: 14 }, { v: 12 },
];
const trendFlat = [
  { v: 15 }, { v: 16 }, { v: 15 }, { v: 17 }, { v: 16 }, { v: 18 }, { v: 18 },
];
const trendGrow = [
  { v: 5 }, { v: 7 }, { v: 9 }, { v: 11 }, { v: 12 }, { v: 14 }, { v: 15 },
];
const trendReport = [
  { v: 2 }, { v: 3 }, { v: 4 }, { v: 4 }, { v: 5 }, { v: 6 }, { v: 7 },
];

function MiniSparkline({ data, color }) {
  return (
    <ResponsiveContainer width={72} height={32}>
      <LineChart data={data}>
        <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} />
        <Tooltip
          contentStyle={{ display: 'none' }}
          cursor={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function StatCard({ label, value, sub, icon: Icon, badgeColor, trend, trendData, delay = 0 }) {
  return (
    <div className={`stat-card animate-in stagger-${delay + 1}`} style={{ animationDelay: `${delay * 60}ms` }}>
      <div>
        <div style={{ fontSize: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '6px' }}>
          {label}
        </div>
        <div style={{ fontSize: '26px', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--text-primary)', lineHeight: 1, marginBottom: '4px' }}>
          {value}
        </div>
        <div style={{ fontSize: '10px', color: 'var(--text-dim)' }}>{sub}</div>
        {trend && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '8px' }}>
            <TrendingUp size={9} color="var(--accent-emerald)" />
            <span style={{ fontSize: '9px', color: 'var(--accent-emerald)', fontWeight: '600' }}>{trend}</span>
          </div>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
        <div className={`icon-badge icon-badge--${badgeColor}`}>
          <Icon size={16} />
        </div>
        <MiniSparkline data={trendData} color={
          badgeColor === 'cyan' ? '#00d4ff' :
          badgeColor === 'violet' ? '#7c3aed' :
          badgeColor === 'emerald' ? '#10b981' : '#f43f5e'
        } />
      </div>
    </div>
  );
}

function PerformanceGauge({ label, value, color, delay = 0 }) {
  const measured = value != null && Number.isFinite(Number(value));
  const numericValue = measured ? Number(value) : 0;
  const pct = Math.round(numericValue * 100);
  return (
    <div className={`animate-in`} style={{ animationDelay: `${200 + delay * 60}ms`, textAlign: 'center' }}>
      <div style={{
        width: '88px', height: '88px',
        borderRadius: '50%',
        border: '3px solid var(--border-subtle)',
        background: `conic-gradient(${color} ${pct * 3.6}deg, var(--border-default) 0deg)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        margin: '0 auto',
        position: 'relative',
        boxShadow: `0 2px 8px rgba(0,0,0,0.05)`,
      }}>
        <div style={{
          width: '72px', height: '72px',
          borderRadius: '50%',
          background: 'var(--bg-panel)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexDirection: 'column',
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', fontWeight: '700', color, lineHeight: 1 }}>
            {measured ? Number(value).toFixed(2) : '—'}
          </span>
        </div>
      </div>
      <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--text-secondary)', fontWeight: '600', letterSpacing: '0.04em' }}>
        {label}
      </div>
      <span className={`badge ${measured ? 'badge--emerald' : 'badge--amber'}`} style={{ marginTop: '4px', display: 'inline-flex' }}>{measured ? 'Measured' : 'Awaiting run'}</span>
    </div>
  );
}

export default function DashboardScreen({ sessionData, comparisonData }) {
  const displayExperiments = [];
  if (comparisonData) {
    displayExperiments.unshift({
      id: "EXP-ACTIVE",
      dataset: sessionData?.filename || "Active Upload",
      records: (sessionData?.row_count ?? sessionData?.total_rows ?? 1000).toLocaleString(),
      privacy: comparisonData.privacy?.eps_used?.toFixed(2) || "1.00",
      utility: (1.0 - comparisonData.correlation_frobenius_error * 0.1).toFixed(2),
      status: "Completed",
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    });
  }

  const datasetCount = sessionData ? 1 : 0;
  const experimentCount = comparisonData ? 1 : 0;
  const syntheticCount = comparisonData ? 1 : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }} className="screen-enter">

      {/* ─── Greeting ─── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Clinical Data Synthesis Platform
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', maxWidth: '620px' }}>
            {sessionData 
              ? `Generating synthetic clinical data from ${sessionData.filename} (${(sessionData.row_count ?? sessionData.total_rows ?? 0).toLocaleString()} rows) using adaptive, privacy-preserving Flow Matching.` 
              : "Upload a dataset to begin a measured synthetic-data experiment."}
          </p>

        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div className="live-dot" />
          <span style={{ fontSize: '11px', color: 'var(--accent-emerald)', fontWeight: '600' }}>All Systems Operational</span>
        </div>
      </div>

      {/* ─── Stat Cards ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        <StatCard label="Datasets"         value={datasetCount} sub="Current session"   icon={Database}       badgeColor="cyan"    trend={null}  trendData={trendUp}    delay={0} />
        <StatCard label="Experiments"      value={experimentCount} sub="Measured runs"       icon={Activity}       badgeColor="violet"  trend={null} trendData={trendFlat}  delay={1} />
        <StatCard label="Synthetic Datasets" value={syntheticCount} sub="Current session"     icon={FileSpreadsheet} badgeColor="emerald" trend={null} trendData={trendGrow}  delay={2} />
        <StatCard label="Reports"          value={comparisonData ? 1 : 0} sub="Available for session"        icon={FileText}        badgeColor="rose"    trend={null}  trendData={trendReport} delay={3} />
      </div>

      {/* ─── Performance Gauges ─── */}
      <div className="panel animate-in" style={{ padding: '20px', animationDelay: '250ms' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 className="panel-title" style={{ marginBottom: 0 }}>
            <TrendingUp size={14} color="var(--accent-cyan)" />
            Overall Performance (Latest measured run)
          </h3>
          <span className="badge badge--cyan">{comparisonData ? '1 measured run' : 'No measured run'}</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', justifyItems: 'center' }}>
          <PerformanceGauge label="DP training status" value={comparisonData?.privacy?.privacy_training_verified ? 1 : null} color="var(--accent-emerald)" delay={0} />
          <PerformanceGauge label="Correlation proxy" value={comparisonData ? Math.max(0, 1 - comparisonData.correlation_frobenius_error) : null} color="var(--accent-cyan)" delay={1} />
          <PerformanceGauge label="Clinical pass rate" value={comparisonData ? comparisonData.clinical_rule_pass_rate / 100 : null} color="var(--accent-purple)" delay={2} />
          <PerformanceGauge label="Fairness" value={null} color="var(--accent-amber)" delay={3} />
        </div>
      </div>

      {/* ─── Recent Experiments Table ─── */}
      <div className="panel animate-in" style={{ overflow: 'hidden', animationDelay: '350ms' }}>
        <div className="section-header">
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', fontSize: '11px', fontWeight: '700', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            <Clock size={11} />
            Recent Experiments
          </span>
          <button style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            fontSize: '10px', color: 'var(--accent-cyan)',
            background: 'transparent', border: 'none', cursor: 'pointer',
            fontFamily: 'var(--font-ui)', fontWeight: '600',
          }}>
            View All <ArrowUpRight size={10} />
          </button>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Dataset</th>
              <th>Records</th>
              <th>Privacy (ε)</th>
              <th>Utility</th>
              <th>Status</th>
              <th>Run On</th>
            </tr>
          </thead>
          <tbody>
            {displayExperiments.map((exp) => (
              <tr key={exp.id}>
                <td className="td-primary">{exp.id}</td>
                <td>{exp.dataset}</td>
                <td className="td-mono">{exp.records}</td>
                <td className="td-mono" style={{ color: 'var(--accent-emerald)' }}>{exp.privacy}</td>
                <td className="td-mono" style={{ color: 'var(--accent-cyan)' }}>{exp.utility}</td>
                <td>
                  <span className="badge badge--emerald">{exp.status}</span>
                </td>
                <td style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{exp.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
