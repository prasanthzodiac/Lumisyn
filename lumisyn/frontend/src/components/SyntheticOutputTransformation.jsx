import React, { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Database, LockKeyhole, WandSparkles } from 'lucide-react';

const MAX_PREVIEW_ROWS = 8;

function displayValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

/**
 * Shows the actual generated release rows exactly once as they pass through
 * the final decode/repair/release handoff. It intentionally does not invent
 * moving values or loop a finished dataset forever.
 */
export default function SyntheticOutputTransformation({ preview = [], columns = [], compact = false }) {
  const rows = useMemo(() => Array.isArray(preview) ? preview.slice(0, MAX_PREVIEW_ROWS) : [], [preview]);
  const inferredColumns = useMemo(() => {
    if (columns.length > 0) return columns;
    return rows.length > 0 ? Object.keys(rows[0]) : [];
  }, [columns, rows]);
  const visibleColumns = inferredColumns.slice(0, 6);
  const [visibleRows, setVisibleRows] = useState(0);
  const [phase, setPhase] = useState(rows.length > 0 ? 'decoding' : 'waiting');

  useEffect(() => {
    setVisibleRows(0);
    if (rows.length === 0) {
      setPhase('waiting');
      return undefined;
    }

    setPhase('decoding');
    let nextRow = 0;
    const timer = window.setInterval(() => {
      nextRow += 1;
      setVisibleRows(nextRow);
      if (nextRow >= rows.length) {
        window.clearInterval(timer);
        setPhase('released');
      } else if (nextRow >= Math.ceil(rows.length / 2)) {
        setPhase('repairing');
      }
    }, 180);

    return () => window.clearInterval(timer);
  }, [rows]);

  const phaseLabel = {
    waiting: 'Awaiting generated release rows',
    decoding: 'Decoding latent trajectories',
    repairing: 'Applying measured constraint repair',
    released: 'Synthetic release rows ready',
  }[phase];

  const phaseIcon = phase === 'released' ? CheckCircle2 : phase === 'repairing' ? LockKeyhole : phase === 'decoding' ? WandSparkles : Database;
  const PhaseIcon = phaseIcon;
  const progress = rows.length === 0 ? 0 : Math.round((visibleRows / rows.length) * 100);

  return (
    <div className={`synthetic-output${compact ? ' synthetic-output--compact' : ''}`}>
      <div className="synthetic-output__header">
        <div>
          <div className="synthetic-output__eyebrow">Stage 9 · Release transformation</div>
          <div className="synthetic-output__title">
            <PhaseIcon size={14} />
            {phaseLabel}
          </div>
        </div>
        <span className={`badge ${phase === 'released' ? 'badge--emerald' : 'badge--cyan'}`}>
          {rows.length > 0 ? `${visibleRows}/${rows.length} rows` : 'Measured output pending'}
        </span>
      </div>

      <div className="synthetic-output__flow" aria-label="Stage 9 transformation path">
        <span>latent trajectory</span>
        <span className="synthetic-output__arrow">→</span>
        <span>decoder</span>
        <span className="synthetic-output__arrow">→</span>
        <span>constraint repair</span>
        <span className="synthetic-output__arrow">→</span>
        <strong>released synthetic row</strong>
      </div>

      <div className="progress-track synthetic-output__progress" aria-label={`Stage 9 output progress ${progress}%`}>
        <div className="progress-fill progress-fill--emerald" style={{ width: `${progress}%` }} />
      </div>

      {rows.length === 0 ? (
        <div className="synthetic-output__empty">
          The final table will appear here from the backend-generated synthetic release. No placeholder values are shown.
        </div>
      ) : (
        <div className="synthetic-output__table-wrap">
          <table className="synthetic-output__table">
            <thead>
              <tr>
                <th>#</th>
                {visibleColumns.map(column => <th key={column}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, visibleRows).map((row, index) => (
                <tr key={`${index}-${visibleColumns.join('|')}`} className="synthetic-output__row">
                  <td>{index + 1}</td>
                  {visibleColumns.map(column => <td key={column}>{displayValue(row[column])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
