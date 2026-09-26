import React, { useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { PieChart as PieIcon, Activity, Info } from 'lucide-react';

const TABS = ['Overview', 'Distribution', 'Correlation', 'Missing Values', 'Target'];
const EMPTY_FEATURES = [];
const tooltipStyle = { contentStyle: { background: 'var(--bg-raised)', border: '1px solid var(--border-default)', borderRadius: '8px', fontSize: '10px', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' } };

export default function DataVisualizationScreen({ comparisonData }) {
  const [tab, setTab] = useState('Overview');
  const features = comparisonData?.numeric_features ?? EMPTY_FEATURES;

  const chartSets = useMemo(() => features.slice(0, 3).map((feature, index) => {
    const histogram = comparisonData?.histograms?.[feature];
    return {
      feature,
      color: ['var(--accent-cyan)', 'var(--accent-purple)', 'var(--accent-emerald)'][index],
      data: histogram ? histogram.bin_centers.map((bin, i) => ({ bin, Real: histogram.real_counts[i] || 0, Synthetic: histogram.syn_counts[i] || 0 })) : [],
    };
  }), [comparisonData, features]);

  const correlationPairs = useMemo(() => {
    const matrix = comparisonData?.real_correlation || {};
    const pairs = [];
    features.forEach((left, i) => features.slice(i + 1).forEach(right => {
      const value = Number(matrix[left]?.[right] ?? matrix[right]?.[left]);
      if (Number.isFinite(value)) pairs.push({ label: `${left} ↔ ${right}`, value, abs: Math.abs(value) });
    }));
    return pairs.sort((a, b) => b.abs - a.abs).slice(0, 5);
  }, [comparisonData, features]);

  if (!comparisonData) {
    return <div className="screen-enter"><div className="panel" style={{ padding: '30px', textAlign: 'center' }}><Info size={26} color="var(--text-muted)" style={{ marginBottom: '8px' }} /><h3 style={{ fontSize: '15px' }}>No measured visualization data</h3><p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Generate synthetic data first. Charts will be populated from the backend comparison response.</p></div></div>;
  }

  const ratio = comparisonData.class_ratio_summary;
  const targetData = ratio ? [{ name: 'Real positive', value: ratio.real_positive_ratio * 100, color: 'var(--accent-cyan)' }, { name: 'Synthetic positive', value: ratio.syn_positive_ratio * 100, color: 'var(--accent-purple)' }] : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">
      <div className="pill-tabs">{TABS.map(item => <button key={item} onClick={() => setTab(item)} className={`pill-tab ${tab === item ? 'pill-tab--active' : ''}`}>{item}</button>)}</div>
      <div className="panel" style={{ padding: '12px 14px', fontSize: '11px', color: 'var(--text-secondary)' }}><strong style={{ color: 'var(--text-primary)' }}>{tab}</strong> · measured from {comparisonData.dataset_name}; numeric features: {features.length}</div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
        {chartSets.length === 0 ? <div className="panel" style={{ padding: '20px', gridColumn: '1 / -1', color: 'var(--text-muted)' }}>No numeric feature histograms were returned by the comparison service.</div> : chartSets.map(chart => (
          <div key={chart.feature} className="panel" style={{ padding: '14px' }}><h4 className="panel-title">{chart.feature} distribution</h4><div style={{ height: '160px' }}><ResponsiveContainer width="100%" height="100%"><BarChart data={chart.data}><XAxis dataKey="bin" stroke="var(--text-dim)" fontSize={9} /><YAxis stroke="var(--text-dim)" fontSize={9} /><Tooltip {...tooltipStyle} /><Bar dataKey="Real" fill={chart.color} opacity={0.75} /><Bar dataKey="Synthetic" fill="var(--accent-rose)" opacity={0.65} /></BarChart></ResponsiveContainer></div></div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div className="panel" style={{ padding: '16px' }}><h4 className="panel-title"><PieIcon size={13} color="var(--accent-purple)" /> Target positive-rate comparison</h4>{targetData.length === 0 ? <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>No target ratio returned.</div> : <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}><div style={{ width: '140px', height: '140px' }}><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={targetData} innerRadius={42} outerRadius={58} dataKey="value">{targetData.map(entry => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip {...tooltipStyle} /></PieChart></ResponsiveContainer></div><div>{targetData.map(entry => <div key={entry.name} style={{ fontSize: '11px', marginBottom: '8px', color: entry.color }}>{entry.name}: {entry.value.toFixed(2)}%</div>)}</div></div>}</div>
        <div className="panel" style={{ padding: '16px' }}><h4 className="panel-title"><Activity size={13} color="var(--accent-cyan)" /> Highest measured correlations</h4>{correlationPairs.length === 0 ? <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>No correlation pairs returned.</div> : correlationPairs.map(pair => <div key={pair.label} style={{ marginBottom: '10px' }}><div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}><span>{pair.label}</span><span style={{ fontFamily: 'var(--font-mono)' }}>{pair.value.toFixed(3)}</span></div><div className="progress-track" style={{ marginTop: '4px' }}><div className="progress-fill" style={{ width: `${Math.min(100, pair.abs * 100)}%`, background: 'var(--accent-cyan)' }} /></div></div>)}</div>
      </div>
    </div>
  );
}
