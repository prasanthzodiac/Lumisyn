import React from 'react';
import { 
  LayoutDashboard, Database, GitBranch, BarChart3, 
  FileText, ShieldCheck, Layers, 
  Lock, Play, FileSpreadsheet, ChevronRight, Cpu
} from 'lucide-react';

const navGroups = [
  {
    label: 'Core',
    items: [
      { id: 'dashboard',   num: '01', label: 'Dashboard',         icon: LayoutDashboard },
      { id: 'upload',      num: '02', label: 'Upload Dataset',    icon: Database },
      { id: 'intelligence',num: '03', label: 'Dataset Intelligence',icon: Cpu },
      { id: 'datavis',     num: '04', label: 'Data Visualization',icon: BarChart3 },
    ]
  },
  {
    label: 'Analysis',
    items: [
      { id: 'graph',       num: '05', label: 'Clinical KG',       icon: GitBranch },
      { id: 'pipeline',    num: '06', label: 'Pipeline Overview', icon: Layers },
      { id: 'stage_hvae',  num: '07', label: 'Stage: HVAE',       icon: Layers },
      { id: 'privacy',     num: '08', label: 'Privacy Controller',icon: Lock },
    ]
  },
  {
    label: 'Validation',
    items: [
      { id: 'flow_anim',   num: '09', label: 'Flow Matching',     icon: Play },
      { id: 'comparison',  num: '10', label: 'Orig vs Synthetic', icon: BarChart3 },
      { id: 'validation',  num: '11', label: 'Statistical Valid.',icon: ShieldCheck },
      { id: 'consistency', num: '12', label: 'Clinical Consistency',icon: ShieldCheck },
    ]
  },
  {
    label: 'Reports',
    items: [
      { id: 'history',     num: '13', label: 'Experiment History',icon: FileSpreadsheet },
      { id: 'reports',     num: '14', label: 'Reports & Downloads',icon: FileText },
    ]
  }
];

export default function Sidebar({ currentScreen, setCurrentScreen, sessionData }) {
  return (
    <aside className="sidebar">

      {/* ─── Brand ─── */}
      <div>
        <div style={{ 
          padding: '20px 16px 14px',
          borderBottom: '1px solid var(--border-default)',
        }}>
          {/* Logo Mark */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <div style={{
              width: '36px', height: '36px',
              borderRadius: '8px',
              background: '#ffffff',
              border: '1px solid var(--border-default)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 2px 4px rgba(0,0,0,0.02)',
              flexShrink: 0,
              position: 'relative'
            }}>
              <span style={{ fontFamily: 'Georgia, serif', fontSize: '20px', fontWeight: 'bold', color: 'var(--text-primary)', marginTop: '-2px' }}>L</span>
              <span style={{ position: 'absolute', top: '8px', right: '5px', fontSize: '9px', color: 'var(--text-primary)', opacity: 0.9 }}>✦</span>
            </div>
            <div>
              <div style={{
                fontSize: '14px', fontWeight: '800',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-ui)',
                letterSpacing: '0.05em',
                textTransform: 'uppercase'
              }}>
                Lumisyn
              </div>
              <div style={{ 
                fontSize: '9px', 
                color: 'var(--text-secondary)', 
                letterSpacing: '0.12em', 
                textTransform: 'uppercase',
                fontWeight: '600',
                marginTop: '-2px'
              }}>
                Lab
              </div>
            </div>
          </div>

          <div style={{ 
            fontSize: '8px', 
            color: 'var(--text-muted)', 
            letterSpacing: '0.04em', 
            textTransform: 'uppercase', 
            fontWeight: '700',
            marginBottom: '10px'
          }}>
            Synthetic Healthcare Intelligence
          </div>

        </div>

        {/* ─── Active Dataset Info Card ─── */}
        {sessionData && (
          <div style={{
            margin: '12px 12px 6px',
            padding: '10px 12px',
            background: 'rgba(99, 102, 241, 0.04)',
            border: '1px solid rgba(99, 102, 241, 0.08)',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px'
          }}>
            <div style={{ fontSize: '8px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
              Viewing Dataset
            </div>
            <div style={{ 
              fontSize: '11px', 
              fontWeight: '700', 
              color: 'var(--text-primary)', 
              overflow: 'hidden', 
              textOverflow: 'ellipsis', 
              whiteSpace: 'nowrap' 
            }} title={sessionData.filename}>
              {sessionData.filename}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
              <span className="badge badge--emerald" style={{ fontSize: '8px', padding: '1px 5px' }}>
                {(sessionData.row_count ?? sessionData.total_rows ?? 0).toLocaleString()} rows
              </span>
              <span className="badge badge--violet" style={{ fontSize: '8px', padding: '1px 5px' }}>
                {sessionData.column_count ?? sessionData.total_columns ?? 0} cols
              </span>
            </div>

          </div>
        )}

        {/* ─── Navigation ─── */}
        <nav style={{ 
          padding: '8px 8px', 
          overflowY: 'auto', 
          maxHeight: 'calc(100vh - 180px)',
          scrollbarWidth: 'thin',
        }}>
          {navGroups.map((group) => (
            <div key={group.label} style={{ marginBottom: '4px' }}>
              <div className="nav-group-label">{group.label}</div>
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = currentScreen === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentScreen(item.id)}
                    className={`sidebar-nav-item ${isActive ? 'sidebar-nav-item--active' : ''}`}
                  >
                    <Icon size={13} style={{ flexShrink: 0, opacity: isActive ? 1 : 0.5 }} />
                    <span className="nav-label" style={{ fontSize: '11px' }}>{item.label}</span>
                    {isActive && (
                      <ChevronRight size={11} style={{ color: 'var(--accent-cyan)', flexShrink: 0, opacity: 0.7 }} />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>
      </div>
    </aside>
  );
}
