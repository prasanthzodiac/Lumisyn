import React, { useState, useEffect } from 'react';
import { Play, Pause, Activity, Zap, Info } from 'lucide-react';

const flowSteps = [
  { t: "0.00", label: "Prior Noise x₀", bg: 'rgba(244,63,94,0.15)',    border: 'rgba(244,63,94,0.35)',    dot: 'var(--accent-rose)',    textColor: 'var(--accent-rose)'    },
  { t: "0.25", label: "Early Flow",     bg: 'rgba(245,158,11,0.12)',   border: 'rgba(245,158,11,0.3)',    dot: 'var(--accent-amber)',   textColor: 'var(--accent-amber)'   },
  { t: "0.50", label: "Mid Flow",       bg: 'rgba(16,185,129,0.12)',   border: 'rgba(16,185,129,0.3)',    dot: 'var(--accent-emerald)', textColor: 'var(--accent-emerald)' },
  { t: "0.75", label: "Late Flow",      bg: 'rgba(0,212,255,0.12)',    border: 'rgba(0,212,255,0.35)',    dot: 'var(--accent-cyan)',    textColor: 'var(--accent-cyan)'    },
  { t: "1.00", label: "Synthetic x₁",  bg: 'rgba(168,85,247,0.12)',   border: 'rgba(168,85,247,0.35)',   dot: 'var(--accent-purple)',  textColor: 'var(--accent-purple)'  },
];

export default function FlowMatchingAnimationScreen() {
  const [timeStep, setTimeStep] = useState(0.75);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    let interval;
    if (playing) {
      interval = setInterval(() => {
        setTimeStep(prev => prev >= 1.0 ? 0.0 : parseFloat((prev + 0.05).toFixed(2)));
      }, 300);
    }
    return () => clearInterval(interval);
  }, [playing]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }} className="screen-enter">

      {/* ─── Header ─── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Activity size={16} color="var(--accent-cyan)" />
            Stage 6: Continuous OT-CFM Particle Flow Simulation
          </h2>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Continuous velocity field regression v_θ(x_t, t, c) along linear Optimal Transport trajectories.
          </p>
        </div>
        <button
          onClick={() => setPlaying(!playing)}
          className="btn-primary"
          style={{ flexShrink: 0 }}
        >
          {playing ? <Pause size={13} /> : <Play size={13} />}
          {playing ? 'Pause Simulation' : 'Play Simulation'}
        </button>
      </div>

      {/* ─── Timeline ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
        {flowSteps.map((step) => {
          const isActive = Math.abs(parseFloat(step.t) - timeStep) < 0.15;
          return (
            <div key={step.t} style={{
              padding: '16px',
              borderRadius: 'var(--radius-lg)',
              background: isActive ? step.bg : 'var(--bg-panel)',
              border: `1px solid ${isActive ? step.border : 'var(--border-subtle)'}`,
              textAlign: 'center',
              transition: 'all var(--transition-med)',
              transform: isActive ? 'translateY(-3px) scale(1.02)' : 'none',
              boxShadow: isActive ? `0 8px 24px ${step.bg}` : 'none',
            }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: isActive ? step.textColor : 'var(--text-dim)', fontWeight: '600', marginBottom: '10px' }}>
                t = {step.t}
              </div>
              {/* Animated particle dot */}
              <div style={{
                width: '48px', height: '48px',
                borderRadius: '50%',
                background: isActive ? step.bg : 'rgba(255,255,255,0.04)',
                border: `2px solid ${isActive ? step.border : 'var(--border-subtle)'}`,
                margin: '0 auto 10px',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: isActive ? `0 0 16px ${step.dot}44` : 'none',
                transition: 'all var(--transition-med)',
              }}>
                <div style={{
                  width: '12px', height: '12px',
                  borderRadius: '50%',
                  background: isActive ? step.dot : 'var(--text-dim)',
                  boxShadow: isActive ? `0 0 8px ${step.dot}` : 'none',
                  animation: isActive ? 'glowPulse 1.5s infinite' : 'none',
                }} />
              </div>
              <div style={{ fontSize: '11px', fontWeight: '600', color: isActive ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {step.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* ─── Progress Bar ─── */}
      <div className="panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: '600' }}>Trajectory Progress</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-cyan)', fontWeight: '700' }}>
            t = {timeStep.toFixed(2)}
          </span>
        </div>
        <div className="progress-track" style={{ height: '6px' }}>
          <div className="progress-fill" style={{ width: `${timeStep * 100}%` }} />
        </div>
      </div>

      {/* ─── Info Grid ─── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
        {[
          { label: 'Generative Model',  value: 'Constraint-Guided OT-CFM', sub: 'Continuous Vector Field', icon: Activity, color: 'cyan' },
          { label: 'ODE Integration',   value: '100 Euler Steps',          sub: 'Adaptive Step Controller', icon: Zap, color: 'emerald' },
          { label: 'Active Constraints',value: 'Clinical + Privacy Rules', sub: 'Stage 7 Repair Active',    icon: Info, color: 'violet' },
        ].map((info, i) => {
          const Icon = info.icon;
          return (
            <div key={i} className="panel animate-in" style={{ padding: '14px', animationDelay: `${i * 50}ms`, display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div className={`icon-badge icon-badge--${info.color}`}>
                <Icon size={14} />
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '2px' }}>{info.label}</div>
                <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)' }}>{info.value}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: '1px' }}>{info.sub}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
