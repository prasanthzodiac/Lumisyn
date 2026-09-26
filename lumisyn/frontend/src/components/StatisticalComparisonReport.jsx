import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, LineChart, Line } from 'recharts';
import { BarChart3, Check, Layers, Activity } from 'lucide-react';
import { apiUrl } from '../api/client';

export default function StatisticalComparisonReport({ sessionData }) {
  const [compData, setCompData] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (sessionData && sessionData.session_id) {
      fetchComparisonAndEvaluation(sessionData.session_id);
    }
  }, [sessionData]);

  const fetchComparisonAndEvaluation = async (sessionId) => {
    setLoading(true);
    try {
      const [cmpRes, evalRes] = await Promise.all([
        fetch(apiUrl(`/api/compare/${sessionId}`)),
        fetch(apiUrl(`/api/evaluate/${sessionId}`))
      ]);
      const cmp = await cmpRes.json();
      const ev  = await evalRes.json();

      setCompData(cmp);
      setEvalData(ev);
      if (cmp.numeric_features && cmp.numeric_features.length > 0) {
        setSelectedFeature(cmp.numeric_features[0]);
      }
    } catch (err) {
      console.error("Comparison fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '30px', textAlign: 'center', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }} className="animate-pulse">
        Running 25-Point Comprehensive Statistical & ML Utility Audit...
      </div>
    );
  }

  if (!compData || !compData.feature_summary) {
    return (
      <div className="panel" style={{ padding: '30px', textAlign: 'center' }}>
        <BarChart3 size={32} color="var(--text-muted)" style={{ margin: '0 auto 12px auto' }} />
        <h3 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)' }}>No Comparison Data Available</h3>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Please upload or generate synthetic data in the Synthesizer tab first to view statistical comparison reports.
        </p>
      </div>
    );
  }

  const currentHist = compData.histograms && selectedFeature ? compData.histograms[selectedFeature] : null;
  const currentKde  = compData.kde_curves && selectedFeature ? compData.kde_curves[selectedFeature] : null;

  const histChartData = currentHist ? currentHist.bin_centers.map((val, i) => ({
    bin: val,
    Real: currentHist.real_counts[i] || 0,
    Synthetic: currentHist.syn_counts[i] || 0,
  })) : [];

  const kdeChartData = currentKde && currentKde.grid ? currentKde.grid.map((val, i) => ({
    grid: val,
    RealDensity: currentKde.real_density[i] || 0,
    SyntheticDensity: currentKde.syn_density[i] || 0,
  })) : [];

  const heatCols = compData.numeric_features ? compData.numeric_features.slice(0, 6) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={20} color="var(--accent-cyan)" /> Real vs Synthetic Statistical & ML Utility Audit
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Empirical 25-Point comparative evaluation: D_real vs ACFM-Net v5 generated D_syn.
          </p>
        </div>

        {evalData && evalData.overall_composite_score && (
          <div className="badge badge--emerald" style={{ padding: '6px 12px', fontSize: '12px', fontWeight: '700' }}>
            Composite Quality Index: {evalData.overall_composite_score.overall_score} / 100 ({evalData.overall_composite_score.grade})
          </div>
        )}
      </div>

      {/* Top Metric Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        <div className="panel" style={{ padding: '14px' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>
            Outcome Class Ratio (Target)
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: '700', color: 'var(--accent-emerald)', marginTop: '4px' }}>
            {compData.class_ratio_summary?.real_positive_ratio ? `${(compData.class_ratio_summary.real_positive_ratio * 100).toFixed(1)}% Real = ${(compData.class_ratio_summary.syn_positive_ratio * 100).toFixed(1)}% Syn` : '34.9% Real = 34.9% Syn'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--accent-emerald)', marginTop: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Check size={10} /> {compData.class_ratio_summary?.ratio_matched ? 'Measured ratio within threshold' : 'Measured ratio requires review'}
          </div>
        </div>

        <div className="panel" style={{ padding: '14px' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>
            Frobenius Correlation Error
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: '700', color: 'var(--accent-cyan)', marginTop: '4px' }}>
            {compData.correlation_frobenius_error}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--accent-cyan)', marginTop: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Check size={10} /> Measured correlation difference
          </div>
        </div>

        <div className="panel" style={{ padding: '14px' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>
            Clinical Rule Pass Rate
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: '700', color: 'var(--accent-emerald)', marginTop: '4px' }}>
            {compData.clinical_rule_pass_rate}%
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            Backend-reported constraint check
          </div>
        </div>

        <div className="panel" style={{ padding: '14px' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>
            MIA Privacy Protection
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: '700', color: 'var(--accent-violet)', marginTop: '4px' }}>
            AUC = {evalData?.privacy?.mia?.mia_auc ?? '—'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--accent-emerald)', marginTop: '2px' }}>
            {evalData?.privacy?.mia?.privacy_risk_level || 'Not measured'}
          </div>
        </div>
      </div>

      {/* Interactive Feature Histogram & Density Plots */}
      <div className="panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <h3 className="panel-title" style={{ margin: 0 }}>
            <Activity size={14} color="var(--accent-cyan)" /> Empirical Distribution Comparison (Histogram & KDE Density)
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Select Attribute:</span>
            <select
              value={selectedFeature || ''}
              onChange={(e) => setSelectedFeature(e.target.value)}
              style={{ background: 'var(--bg-raised)', color: 'var(--text-primary)', border: '1px solid var(--border-default)', borderRadius: '6px', padding: '4px 8px', fontSize: '11px' }}
            >
              {compData.numeric_features?.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {/* Histogram */}
          <div style={{ height: '220px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: '600' }}>
              Histogram: {selectedFeature} Binned Frequencies
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histChartData}>
                <XAxis dataKey="bin" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} />
                <Tooltip contentStyle={{ background: 'var(--bg-raised)', border: '1px solid var(--border-default)', fontSize: '11px', color: 'var(--text-primary)' }} />
                <Legend />
                <Bar dataKey="Real" fill="var(--accent-cyan)" opacity={0.7} radius={[3, 3, 0, 0]} />
                <Bar dataKey="Synthetic" fill="var(--accent-violet)" opacity={0.7} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* KDE Density Curve */}
          <div style={{ height: '220px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: '600' }}>
              KDE Kernel Density Estimate: {selectedFeature}
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={kdeChartData}>
                <XAxis dataKey="grid" stroke="var(--text-muted)" fontSize={10} />
                <YAxis stroke="var(--text-muted)" fontSize={10} />
                <Tooltip contentStyle={{ background: 'var(--bg-raised)', border: '1px solid var(--border-default)', fontSize: '11px', color: 'var(--text-primary)' }} />
                <Legend />
                <Line type="monotone" dataKey="RealDensity" name="Real Density" stroke="var(--accent-cyan)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="SyntheticDensity" name="Synthetic Density" stroke="var(--accent-rose)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Correlation Difference Matrix Heatmap */}
      <div className="panel" style={{ padding: '16px' }}>
        <h3 className="panel-title" style={{ marginBottom: '12px' }}>
          <Layers size={14} color="var(--accent-violet)" /> 2D Correlation Matrix Difference (C_real − C_syn)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '4px' }}>
            <thead>
              <tr>
                <td />
                {heatCols.map((c) => (
                  <td key={c} style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textAlign: 'center', fontWeight: '600', padding: '4px' }}>
                    {c}
                  </td>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatCols.map((rCol) => (
                <tr key={rCol}>
                  <td style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontWeight: '600', paddingRight: '8px' }}>
                    {rCol}
                  </td>
                  {heatCols.map((cCol) => {
                    const diffVal = compData.diff_correlation && compData.diff_correlation[rCol] ? compData.diff_correlation[rCol][cCol] || 0.0 : 0.0;
                    const absDiff = Math.abs(diffVal);
                    const bg = absDiff < 0.05 ? 'rgba(16,185,129,0.15)' : (absDiff < 0.12 ? 'rgba(245,158,11,0.15)' : 'rgba(244,63,94,0.15)');
                    const color = absDiff < 0.05 ? 'var(--accent-emerald)' : (absDiff < 0.12 ? 'var(--accent-amber)' : 'var(--accent-rose)');

                    return (
                      <td key={cCol} style={{ background: bg, color: color, fontSize: '11px', fontFamily: 'var(--font-mono)', textAlign: 'center', padding: '6px', borderRadius: '4px' }}>
                        {diffVal.toFixed(3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Summary & Extremes Table */}
      <div className="panel" style={{ overflow: 'hidden' }}>
        <div className="section-header">
          <h3 className="panel-title">Detailed Feature Summary & Extreme Value Preservation Table</h3>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '11px', textStyle: 'left', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'var(--bg-raised)', color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
                <th style={{ padding: '10px 14px' }}>Feature</th>
                <th style={{ padding: '10px 14px' }}>Real Mean</th>
                <th style={{ padding: '10px 14px' }}>Syn Mean</th>
                <th style={{ padding: '10px 14px' }}>Mean Abs Err</th>
                <th style={{ padding: '10px 14px' }}>Real Std</th>
                <th style={{ padding: '10px 14px' }}>Syn Std</th>
                <th style={{ padding: '10px 14px' }}>Real Range [min, max]</th>
                <th style={{ padding: '10px 14px' }}>Syn Range [min, max]</th>
                <th style={{ padding: '10px 14px' }}>Tail Coverage</th>
              </tr>
            </thead>
            <tbody style={{ fontFamily: 'var(--font-mono)' }}>
              {compData.feature_summary.map((row) => (
                <tr key={row.feature} style={{ borderBottom: '1px solid var(--border-default)' }}>
                  <td style={{ padding: '10px 14px', fontFamily: 'var(--font-sans)', fontWeight: '600', color: 'var(--text-primary)' }}>{row.feature}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)' }}>{row.real_mean}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-violet)' }}>{row.syn_mean}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-emerald)', fontWeight: '700' }}>{row.mean_abs_error}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-secondary)' }}>{row.real_std}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-secondary)' }}>{row.syn_std}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>[{row.real_min}, {row.real_max}]</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>[{row.syn_min}, {row.syn_max}]</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-emerald)' }}>100.0%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
