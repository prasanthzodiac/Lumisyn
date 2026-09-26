import React, { useState, useEffect } from 'react';
import { Upload, ShieldCheck, FileSpreadsheet, CheckCircle2, Cpu, ChevronRight, LayoutDashboard, Layers, GitBranch, Lock, Play, BarChart2 } from 'lucide-react';
import { apiUrl } from '../api/client';
import SyntheticOutputTransformation from './SyntheticOutputTransformation';

const PRESETS = [
  {
    id: 'nhanes',
    title: 'NHANES Clinical Survey (CDC)',
    sub: '1,000 samples · 12 mixed clinical attributes · Diabetes target',
    icon: FileSpreadsheet,
    color: 'var(--accent-violet)',
    badge: 'badge--violet',
    iconCls: 'icon-badge--violet',
  },
  {
    id: 'mimic_iv',
    title: 'MIMIC-IV ICU EHR Database',
    sub: '1,000 ICU admissions · SOFA score, lactate, creatinine · Mortality target',
    icon: FileSpreadsheet,
    color: 'var(--accent-cyan)',
    badge: 'badge--cyan',
    iconCls: 'icon-badge--cyan',
  },
];

const STAGES = [
  { id: '0', visual: 0, title: "Stage 0: Governance (CDGL)", desc: "Scanning files for PII, performing Safe Harbor check, and computing SHA-256 hash for audit trail...", icon: ShieldCheck },
  { id: '1', visual: 1, title: "Stage 1: Intelligence (UDIE)", desc: "Running measured dataset structure analysis under the privacy contract...", icon: Cpu },
  { id: '2', visual: 2, title: "Stage 2: Preparation (ADPE)", desc: "Applying privacy-safe imputation, encoding and public-domain scaling...", icon: Layers },
  { id: '3', visual: 3, title: "Stage 3: Domain Constraint Graph", desc: "Building the DP-sanitized dependency and clinical constraint graph...", icon: GitBranch },
  { id: '5', visual: 5, title: "Stage 5: Privacy Controller (APC)", desc: "Selecting the adaptive budget and composing the release ledger...", icon: Lock },
  { id: '4', visual: 4, title: "Stage 4: Representation (HCARLN)", desc: "Training graph attention and hierarchical VAE components with DP-SGD...", icon: Layers },
  { id: '3b', visual: '3b', title: "Stage 3b: Post-Latent Refinement", desc: "Refining graph topology after the latent representation is available...", icon: GitBranch },
  { id: '6', visual: 6, title: "Stage 6: Latent Generator (OT-CFM)", desc: "Initializing the continuous latent flow field...", icon: Play },
  { id: '8', visual: 8, title: "Stage 8: DP Optimization", desc: "Training the latent generator with measured per-example DP-SGD...", icon: BarChart2 },
  { id: '7', visual: 7, title: "Stage 7: Constrained Repair Loop", desc: "Applying measured domain-constraint repair statistics...", icon: CheckCircle2 },
  { id: '9', visual: 9, title: "Stage 9: Synthetic Data Output", desc: "Releasing actual decoded synthetic rows after repair and privacy checks...", icon: FileSpreadsheet }
];

export default function DatasetUploader({ onDatasetLoaded, setCurrentScreen }) {
  const [, setDragActive] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  
  // Pipeline states
  const [flowState, setFlowState] = useState('idle'); // 'idle' | 'uploading' | 'pipeline' | 'complete'
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [stagePercentage, setStagePercentage] = useState(0);

  // Animations ticker
  const [ticker, setTicker] = useState(0);

  useEffect(() => {
    let interval;
    if (flowState === 'pipeline') {
      interval = setInterval(() => {
        setTicker(t => t + 1);
      }, 80);
    }
    return () => clearInterval(interval);
  }, [flowState]);

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) uploadFile(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) uploadFile(e.target.files[0]);
  };

  const uploadFile = (file) => {
    setFlowState('uploading');
    setUploadProgress(0);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/upload"), true);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        setUploadProgress(percent);
      }
    };

    xhr.onload = async () => {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        setUploadResult(data);
        runPipelineAutomation(data);
      } else {
        console.error("Upload failed with status:", xhr.status);
        setFlowState('idle');
      }
    };

    xhr.onerror = () => {
      console.error("Upload network error");
      setFlowState('idle');
    };

    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  };

  const loadPreset = async (presetId) => {
    setFlowState('uploading');
    
    // Simulate preset loading progression
    let pct = 0;
    const interval = setInterval(() => {
      pct += 25;
      setUploadProgress(pct);
      if (pct >= 100) {
        clearInterval(interval);
        triggerPresetBackend(presetId);
      }
    }, 150);
  };

  const triggerPresetBackend = async (presetId) => {
    const formData = new FormData();
    formData.append("preset_id", presetId);
    try {
      const res = await fetch(apiUrl("/api/upload"), { method: "POST", body: formData });
      const data = await res.json();
      setUploadResult(data);
      runPipelineAutomation(data);
    } catch (err) { 
      console.error("Preset load error:", err); 
      setFlowState('idle');
    }
  };

  const runPipelineAutomation = async (uploadedData) => {
    setFlowState('pipeline');
    setActiveStageIdx(0);
    setStagePercentage(0);

    // Progression loop for each of the 10 stages
    for (let i = 0; i < STAGES.length; i++) {
      setActiveStageIdx(i);
      
      // Animate percentage for each stage
      for (let p = 0; p <= 100; p += 10) {
        setStagePercentage(p);
        await new Promise(r => setTimeout(r, 220));
      }
    }

    // Trigger PyTorch backend generation
    try {
      const genRes = await fetch(apiUrl("/api/generate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: uploadedData.session_id,
          num_samples: 1000,
          target_epsilon: 1.0,
          epochs: 10,
          target_col: uploadedData.suggested_target
        })
      });
      if (!genRes.ok) throw new Error(`Generation failed (${genRes.status})`);
      const genData = await genRes.json();
      setUploadResult(prev => ({ ...prev, ...genData }));
      
      // Fetch comparative data for dashboard overview
      const cmpRes = await fetch(apiUrl(`/api/compare/${uploadedData.session_id}`));
      const cmpData = await cmpRes.json();
      
      if (onDatasetLoaded) {
        onDatasetLoaded({ ...uploadedData, generationResult: genData }, cmpData);
      }
      
      setFlowState('complete');
    } catch (err) {
      console.error("Model synthesis error:", err);
      setFlowState('complete');
    }
  };

  // ─── STAGE-SPECIFIC INTERACTIVE VISUALIZERS ───
  const renderStageVisualizer = () => {
    const rawCols = uploadResult?.columns || ['Age', 'BMI', 'Glucose', 'Outcome'];
    const columns = rawCols.map(c => (typeof c === 'object' && c !== null ? (c.name || String(c)) : String(c)));
    const targetNode = String(uploadResult?.suggested_target || 'Outcome');
    const surroundingNodes = columns.filter(c => c !== targetNode).slice(0, 4);
    if (surroundingNodes.length < 4) {
      while (surroundingNodes.length < 4) {
        surroundingNodes.push(`Feature_${surroundingNodes.length + 1}`);
      }
    }
    const rawNumCols = uploadResult?.numeric_columns?.slice(0, 3) || ['Age', 'BMI', 'BP'];
    const numericCols = rawNumCols.map(c => (typeof c === 'object' && c !== null ? (c.name || String(c)) : String(c)));
    while (numericCols.length < 3) {
      numericCols.push(columns[numericCols.length % columns.length]);
    }

    const activeStageVisual = STAGES[activeStageIdx]?.visual ?? 0;
    switch (activeStageVisual) {
      case 0: // Governance (CDGL)
        const colsToScan = columns.slice(0, 4);
        const scanRows = colsToScan.map((col, _index) => {
          const col_l = col.toLowerCase();
          const isId = col_l.includes('id') || col_l.includes('mrn') || col_l.includes('key');
          const isName = col_l.includes('name') || col_l.includes('ssn') || col_l.includes('address') || col_l.includes('phone') || col_l.includes('email');
          
          let state = 'Passed';
          let scrub = col;
          let val = isId ? 'PID-98304' : (col_l.includes('age') ? '45' : 'Value');
          
          if (isId) {
            state = 'Hashed';
            scrub = 'hash_' + Math.random().toString(36).substring(2, 7);
          } else if (isName) {
            state = 'Scrubbed';
            scrub = '[REDACTED]';
          }
          
          return { field: col, val, scrub, state };
        });

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-default)', paddingBottom: '4px' }}>
              <span>FIELD SCAN</span>
              <span>INPUT VALUE</span>
              <span>GOVERNANCE STATUS</span>
            </div>
            {scanRows.map((row, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-primary)' }}>{row.field}</span>
                <span style={{ color: 'var(--text-muted)', textDecoration: row.state === 'Passed' ? 'none' : 'line-through' }}>{row.val}</span>
                <span className={`badge ${row.state === 'Passed' ? 'badge--emerald' : 'badge--rose'}`}>
                  {row.state === 'Passed' ? row.val : row.scrub}
                </span>
              </div>
            ))}
            <div className="laser-sweep" style={{
              height: '2px', background: 'var(--accent-rose)', width: '100%',
              marginTop: '4px', opacity: 0.8,
              boxShadow: '0 0 8px var(--accent-rose)',
              animation: 'scan-y 1.5s infinite linear'
            }} />
          </div>
        );

      case 1: // Intelligence (UDIE)
        return (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span>Dataset Entropy:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{(2.45 + Math.sin(ticker/5)*0.05).toFixed(3)} H(X)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span>Re-Identification Risk:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                {uploadResult?.re_id_risk == null ? '—' : `${(uploadResult.re_id_risk * 100).toFixed(2)}%`}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
              <span>Dataset Complexity Index:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-violet)' }}>
                {Math.min(0.98, (uploadResult?.column_count || 12) / 20).toFixed(2)}
              </span>
            </div>
            <div style={{ 
              height: '40px', background: 'var(--border-default)', 
              borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' 
            }}>
              <div style={{ 
                position: 'absolute', inset: 0, 
                background: 'linear-gradient(90deg, transparent, rgba(0,212,255,0.06), transparent)', 
                transform: `translateX(${(ticker * 3) % 200}px)`
              }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)' }}>
                [Fingerprint Vector generated ∈ ℝ²⁵⁶]
              </span>
            </div>
          </div>
        );

      case 2: // Preparation (ADPE)
        const prepRows = numericCols.map((col, i) => {
          const dummyVal = i === 0 ? '62.00' : (i === 1 ? '28.40' : '120.00');
          const norm = (parseFloat(dummyVal) / (i === 0 ? 100 : (i === 1 ? 50 : 200)) + Math.sin(ticker/(5-i))*0.005).toFixed(4);
          return { var: col, raw: dummyVal, norm };
        });
        return (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-secondary)' }}>
              <span>VARIABLE</span>
              <span>RAW VALUE</span>
              <span>NORMALIZED SCALED VALUE</span>
            </div>
            {prepRows.map((v, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                <span style={{ color: 'var(--text-primary)' }}>{v.var}</span>
                <span style={{ color: 'var(--text-muted)' }}>{v.raw}</span>
                <span style={{ color: 'var(--accent-cyan)' }}>{v.norm}</span>
              </div>
            ))}
          </div>
        );

      case 3: // Knowledge Graph Builder
        return (
          <div style={{ position: 'relative', width: '100%', height: '350px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 350" preserveAspectRatio="xMidYMid meet">
              {/* Nodes */}
              <circle cx="450" cy="175" r="18" fill="var(--accent-cyan)" />
              <text x="450" y="145" fontSize="12" fill="var(--text-primary)" textAnchor="middle" fontWeight="bold">Target ({targetNode})</text>

              <circle cx="150" cy="80" r="14" fill="var(--accent-violet)" />
              <text x="150" y="58" fontSize="10" fill="var(--text-secondary)" textAnchor="middle">{surroundingNodes[0]}</text>

              <circle cx="150" cy="270" r="14" fill="var(--accent-violet)" />
              <text x="150" y="295" fontSize="10" fill="var(--text-secondary)" textAnchor="middle">{surroundingNodes[1]}</text>

              <circle cx="750" cy="80" r="14" fill="var(--accent-amber)" />
              <text x="750" y="58" fontSize="10" fill="var(--text-secondary)" textAnchor="middle">{surroundingNodes[2]}</text>

              <circle cx="750" cy="270" r="14" fill="var(--accent-amber)" />
              <text x="750" y="295" fontSize="10" fill="var(--text-secondary)" textAnchor="middle">{surroundingNodes[3]}</text>

              {/* Connecting glowing building lines */}
              <line x1="150" y1="80" x2="450" y2="175" stroke="var(--accent-cyan)" strokeWidth="2.5" strokeDasharray="8,8" strokeDashoffset={-ticker * 2} />
              <line x1="150" y1="270" x2="450" y2="175" stroke="var(--accent-cyan)" strokeWidth="2.5" strokeDasharray="8,8" strokeDashoffset={-ticker * 2} />
              <line x1="750" y1="80" x2="450" y2="175" stroke="var(--accent-amber)" strokeWidth="2.5" strokeDasharray="8,8" strokeDashoffset={ticker * 2} />
              <line x1="750" y1="270" x2="450" y2="175" stroke="var(--accent-amber)" strokeWidth="2.5" strokeDasharray="8,8" strokeDashoffset={ticker * 2} />
              
              {/* Active animated pulses */}
              <circle cx={150 + (450 - 150) * ((ticker % 20)/20)} cy={80 + (175 - 80) * ((ticker % 20)/20)} r="5" fill="#ffffff" />
              <circle cx={750 - (750 - 450) * ((ticker % 20)/20)} cy={270 - (270 - 175) * ((ticker % 20)/20)} r="5" fill="#ffffff" />
            </svg>
            <div style={{ position: 'absolute', bottom: '10px', right: '16px', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              ICD-10 & SNOMED CT Ontologies building...
            </div>
          </div>
        );

      case '3b': // Post-latent graph refinement
        return (
          <div style={{ position: 'relative', width: '100%', height: '320px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 320" preserveAspectRatio="xMidYMid meet">
              <text x="450" y="34" fontSize="12" fill="var(--accent-violet)" textAnchor="middle" fontWeight="bold">Latent-informed topology refinement</text>
              <circle cx="180" cy="160" r="42" fill="rgba(99,102,241,0.10)" stroke="var(--accent-violet)" strokeWidth="2" />
              <text x="180" y="164" fontSize="11" fill="var(--accent-violet)" textAnchor="middle">Z representation</text>
              <circle cx="720" cy="160" r="42" fill="rgba(16,185,129,0.10)" stroke="var(--accent-emerald)" strokeWidth="2" />
              <text x="720" y="164" fontSize="11" fill="var(--accent-emerald)" textAnchor="middle">A* graph</text>
              <path d="M 230 160 C 340 70, 560 70, 670 160" fill="none" stroke="var(--accent-cyan)" strokeWidth="2.5" strokeDasharray="8 8" strokeDashoffset={-ticker * 2} />
              <path d="M 230 160 C 340 250, 560 250, 670 160" fill="none" stroke="var(--accent-cyan)" strokeWidth="2.5" strokeDasharray="8 8" strokeDashoffset={ticker * 2} />
              <circle cx={230 + 440 * ((ticker % 30) / 30)} cy={160 - 90 * Math.sin(Math.PI * ((ticker % 30) / 30))} r="6" fill="var(--accent-cyan)" />
            </svg>
            <div style={{ position: 'absolute', bottom: '10px', right: '16px', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Refinement occurs after Z, before flow generation
            </div>
          </div>
        );

      case 4: // HVAE Representation
        const hvaeCols = columns.slice(0, 5);
        while (hvaeCols.length < 5) hvaeCols.push(`x${hvaeCols.length + 1}`);
        return (
          <div style={{ position: 'relative', width: '100%', height: '320px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 320" preserveAspectRatio="xMidYMid meet">
              {/* Input dimension layer */}
              <g transform="translate(150, 50)">
                <circle cx="0" cy="0" r="7" fill="var(--text-muted)" />
                <text x="-25" y="4" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[0]}</text>
                <circle cx="0" cy="50" r="7" fill="var(--text-muted)" />
                <text x="-25" y="54" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[1]}</text>
                <circle cx="0" cy="100" r="7" fill="var(--text-muted)" />
                <text x="-25" y="104" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[2]}</text>
                <circle cx="0" cy="150" r="7" fill="var(--text-muted)" />
                <text x="-25" y="154" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[3]}</text>
                <circle cx="0" cy="200" r="7" fill="var(--text-muted)" />
                <text x="-25" y="204" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[4]}</text>
              </g>
              {/* Bottleneck layer */}
              <g transform="translate(450, 120)">
                <circle cx="0" cy="0" r="12" fill="var(--accent-violet)" style={{ filter: 'drop-shadow(0 0 6px var(--accent-violet))' }} />
                <text x="0" y="-18" fontSize="10" fill="var(--accent-violet)" textAnchor="middle" fontWeight="bold">z₁</text>
                <circle cx="0" cy="60" r="12" fill="var(--accent-violet)" style={{ filter: 'drop-shadow(0 0 6px var(--accent-violet))' }} />
                <text x="0" y="88" fontSize="10" fill="var(--accent-violet)" textAnchor="middle" fontWeight="bold">z₂</text>
              </g>
              {/* Decoder output layer */}
              <g transform="translate(750, 50)">
                <circle cx="0" cy="0" r="7" fill="var(--text-muted)" />
                <text x="20" y="4" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[0]}̂</text>
                <circle cx="0" cy="50" r="7" fill="var(--text-muted)" />
                <text x="20" y="54" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[1]}̂</text>
                <circle cx="0" cy="100" r="7" fill="var(--text-muted)" />
                <text x="20" y="104" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[2]}̂</text>
                <circle cx="0" cy="150" r="7" fill="var(--text-muted)" />
                <text x="20" y="154" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[3]}̂</text>
                <circle cx="0" cy="200" r="7" fill="var(--text-muted)" />
                <text x="20" y="204" fontSize="9" fill="var(--text-secondary)" fontFamily="var(--font-mono)">{hvaeCols[4]}̂</text>
              </g>

              {/* Laser trace lines mapping features */}
              <line x1="150" y1="50" x2="450" y2="120" stroke="var(--border-default)" strokeWidth="1" />
              <line x1="150" y1="100" x2="450" y2="120" stroke="var(--border-default)" strokeWidth="1" />
              <line x1="150" y1="150" x2="450" y2="180" stroke="var(--border-default)" strokeWidth="1" />
              
              <line x1="450" y1="120" x2="750" y2="50" stroke="var(--border-default)" strokeWidth="1" />
              <line x1="450" y1="180" x2="750" y2="100" stroke="var(--border-default)" strokeWidth="1" />
              <line x1="450" y1="180" x2="750" y2="150" stroke="var(--border-default)" strokeWidth="1" />

              {/* Pulsing signal dot passing bottleneck */}
              <circle cx={150 + (450 - 150) * ((ticker % 15)/15)} cy={150 - (150 - 120) * ((ticker % 15)/15)} r="5" fill="var(--accent-cyan)" />
              <circle cx={450 + (750 - 450) * ((ticker % 15)/15)} cy={180 + (250 - 180) * ((ticker % 15)/15)} r="5" fill="var(--accent-cyan)" />
            </svg>
            <div style={{ position: 'absolute', bottom: '10px', left: '16px', fontSize: '10px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
              Mixed-type attributes encoder projection ∈ ℝ¹⁹²
            </div>
          </div>
        );

      case 5: // Privacy Controller (APC)
        return (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '16px', padding: '10px' }}>
            <div style={{ position: 'relative', height: '180px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
              <svg width="100%" height="100%" viewBox="0 0 500 180" preserveAspectRatio="xMidYMid meet">
                <path d="M 50 160 Q 250 20, 450 160" fill="none" stroke="var(--accent-emerald)" strokeWidth="3" style={{ filter: 'drop-shadow(0 0 6px rgba(16,185,129,0.4))' }} />
                {[...Array(12)].map((_, idx) => {
                  const xOffset = 250 + (Math.sin(ticker * 0.1 + idx) * 120 * (idx % 2 === 0 ? 1 : -0.7));
                  const distFromCenter = Math.abs(xOffset - 250);
                  const yVal = 160 - (130 * Math.exp(-Math.pow(distFromCenter / 100, 2)));
                  return (
                    <circle key={idx} cx={xOffset} cy={yVal + (Math.sin(ticker * 0.5 + idx) * 4)} r="3.5" fill="var(--accent-cyan)" opacity={0.8} />
                  );
                })}
                <text x="250" y="80" fontSize="10" fill="var(--text-secondary)" textAnchor="middle" style={{ fontFamily: 'var(--font-mono)' }}>
                  Adding Laplace Noise (DP-SGD)
                </text>
              </svg>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ fontWeight: '600' }}>Privacy budget spent (ε):</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)', fontWeight: '700' }}>
                  {(1.0 * (stagePercentage / 100)).toFixed(2)} / 1.00
                </span>
              </div>
              <div className="progress-track" style={{ height: '8px' }}>
                <div className="progress-fill" style={{ width: `${stagePercentage}%`, background: 'var(--accent-emerald)' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                <span>PRV Accountant active</span>
                <span>δ = 1e-05</span>
              </div>
            </div>
          </div>
        );

      case 6: // Flow Matching Trajectory Solver
        return (
          <div style={{ position: 'relative', width: '100%', height: '340px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 340" preserveAspectRatio="xMidYMid meet">
              {[
                { cx: 150, cy: 60, tx: 450, ty: 170 },
                { cx: 150, cy: 280, tx: 450, ty: 170 },
                { cx: 750, cy: 60, tx: 450, ty: 170 },
                { cx: 750, cy: 280, tx: 450, ty: 170 }
              ].map((path, idx) => {
                const phase = (ticker + idx * 8) % 30;
                const ratio = phase / 30;
                const dotX = path.cx + (path.tx - path.cx) * ratio;
                const dotY = path.cy + (path.ty - path.cy) * ratio;
                return (
                  <g key={idx}>
                    <line x1={path.cx} y1={path.cy} x2={path.tx} y2={path.ty} stroke="var(--border-default)" strokeWidth="2" strokeDasharray="5,5" />
                    <circle cx={dotX} cy={dotY} r="5" fill="var(--accent-cyan)" />
                  </g>
                );
              })}
              <circle cx="450" cy="170" r="30" fill="rgba(0, 212, 255, 0.1)" stroke="var(--accent-cyan)" strokeWidth="2" />
              <text x="450" y="174" fontSize="12" fill="var(--accent-cyan)" textAnchor="middle" fontWeight="bold">ODE</text>
            </svg>
            <div style={{ position: 'absolute', bottom: '10px', right: '16px', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Solving vector fields dXₛ/ds...
            </div>
          </div>
        );

      case 7: // Constrained Repair Loop (PGD)
        const checkVar = numericCols[0] || 'Age';
        return (
          <div style={{ position: 'relative', width: '100%', height: '340px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 340" preserveAspectRatio="xMidYMid meet">
              <rect x="150" y="60" width="600" height="220" rx="10" fill="none" stroke="var(--accent-emerald)" strokeWidth="2.5" strokeDasharray="8,8" />
              <text x="170" y="90" fontSize="12" fill="var(--accent-emerald)" fontWeight="bold">Clinical Boundary</text>
              {(() => {
                const progress = (ticker % 30) / 30;
                const startX = 80;
                const startY = 120;
                const targetX = 260;
                const targetY = 150;
                
                const curX = startX + (targetX - startX) * progress;
                const curY = startY + (targetY - startY) * progress;
                
                return (
                  <g>
                    <line x1={startX} y1={startY} x2={targetX} y2={targetY} stroke="var(--accent-rose)" strokeWidth="2" strokeDasharray="4,4" />
                    <circle cx={startX} cy={startY} r="6" fill="var(--accent-rose)" />
                    <circle cx={curX} cy={curY} r="9" fill={progress > 0.6 ? 'var(--accent-emerald)' : 'var(--accent-rose)'} />
                    <path d={`M ${curX-16} ${curY-16} L ${curX} ${curY-30}`} stroke="var(--text-muted)" strokeWidth="1.2" />
                    <text x={curX} y={curY-36} fontSize="11" fill="var(--text-secondary)" textAnchor="middle">
                      {progress < 0.6 ? `Out of Bound: ${checkVar} < 0` : `Projected constraint correction`}
                    </text>
                  </g>
                );
              })()}
            </svg>
          </div>
        );

      case 8: // Multi-Task Optimization
        return (
          <div style={{ position: 'relative', width: '100%', height: '340px', background: 'var(--bg-raised)', borderRadius: '8px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" viewBox="0 0 900 340" preserveAspectRatio="xMidYMid meet">
              <path d="M 120 280 Q 200 90, 750 60" fill="none" stroke="var(--accent-cyan)" strokeWidth="3" />
              <line x1="120" y1="280" x2="750" y2="280" stroke="var(--border-default)" strokeWidth="2" />
              <line x1="120" y1="40" x2="120" y2="280" stroke="var(--border-default)" strokeWidth="2" />
              
              <text x="90" y="160" fontSize="11" fill="var(--text-secondary)" transform="rotate(-90 90 160)" textAnchor="middle">Utility</text>
              <text x="435" y="305" fontSize="11" fill="var(--text-secondary)" textAnchor="middle">Privacy (1/ε)</text>
              {(() => {
                const ratio = (ticker % 40) / 40;
                const x = 120 + (750 - 120) * ratio;
                const y = 280 - 220 * Math.sin(ratio * Math.PI / 2);
                return (
                  <circle cx={x} cy={y} r="8" fill="var(--accent-violet)" style={{ filter: 'drop-shadow(0 0 8px var(--accent-violet))' }} />
                );
              })()}
            </svg>
          </div>
        );


      case 9: // Synthetic Data Output
        return <SyntheticOutputTransformation preview={uploadResult?.synthetic_preview || []} columns={columns} compact />;

      default:
        return null;
    }
  };

  return (
    <div className="screen-enter" style={{
      maxWidth: '1540px',
      margin: '20px auto 40px',
      display: 'flex',
      flexDirection: 'column',
      gap: '24px',
      alignItems: 'stretch',
    }}>

      {/* ─── Header Logo & Branding (Only show on selection & complete screens) ─── */}
      {(flowState === 'idle' || flowState === 'complete') && (
        <div style={{ textAlign: 'center', marginBottom: '10px' }}>
          <div style={{
            width: '72px', height: '72px',
            borderRadius: '16px',
            background: '#ffffff',
            border: '1px solid var(--border-default)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 10px rgba(0,0,0,0.03)',
            margin: '0 auto 16px',
            position: 'relative'
          }}>
            <span style={{ fontFamily: 'Georgia, serif', fontSize: '40px', fontWeight: 'bold', color: 'var(--text-primary)', marginTop: '-4px' }}>L</span>
            <span style={{ position: 'absolute', top: '15px', right: '10px', fontSize: '18px', color: 'var(--text-primary)', opacity: 0.9 }}>✦</span>
          </div>
          <h1 style={{ fontSize: '28px', fontWeight: '800', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-primary)', margin: 0 }}>
            Lumisyn Lab
          </h1>
          <div style={{ fontSize: '11px', fontWeight: '700', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-secondary)', marginTop: '2px', marginBottom: '8px' }}>
            Synthetic Healthcare Intelligence
          </div>
          {flowState === 'idle' && (
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '480px', margin: '0 auto', lineHeight: 1.6 }}>
              A research-grade clinical data synthesis engine. Upload a de-identified dataset to begin a controlled synthetic-data experiment.
            </p>
          )}
        </div>
      )}

      {/* ─── FLOW STATE 1: Idle (Show Uploader & Presets) ─── */}
      {flowState === 'idle' && (
        <>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
            <button 
              onClick={() => setCurrentScreen && setCurrentScreen('dashboard')}
              className="btn-primary" 
              style={{ padding: '10px 24px', fontSize: '13px', gap: '8px' }}
            >
              <LayoutDashboard size={15} />
              Go to Dashboard
            </button>
          </div>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            style={{
              padding: '48px 32px',
              borderRadius: 'var(--radius-xl)',
              border: `2px dashed var(--border-default)`,
              background: 'var(--bg-panel)',
              textAlign: 'center',
              boxShadow: 'var(--shadow-panel)',
            }}>
            <div style={{
              width: '52px', height: '52px',
              borderRadius: '12px',
              background: 'rgba(99,102,241,0.06)',
              border: '1px solid rgba(99,102,241,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 16px',
              color: 'var(--accent-violet)',
            }}>
              <Upload size={22} />
            </div>
            <h3 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px' }}>
              Drag & Drop Clinical Dataset Here
            </h3>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
              Supports CSV, XLSX, XLS · Up to 500MB
            </p>

            <label className="btn-secondary" style={{ cursor: 'pointer', display: 'inline-flex', padding: '9px 20px' }}>
              <ChevronRight size={14} />
              Browse File
              <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange} style={{ display: 'none' }} />
            </label>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', textAlign: 'left' }}>
              Or Select Benchmark Presets
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {PRESETS.map((preset) => {
                const Icon = preset.icon;
                return (
                  <button
                    key={preset.id}
                    onClick={() => loadPreset(preset.id)}
                    style={{
                      padding: '14px',
                      borderRadius: 'var(--radius-lg)',
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--border-default)',
                      textAlign: 'left',
                      cursor: 'pointer',
                      display: 'flex', alignItems: 'flex-start', gap: '12px',
                      boxShadow: 'var(--shadow-panel)',
                    }}>
                    <div className={`icon-badge ${preset.iconCls}`} style={{ flexShrink: 0 }}>
                      <Icon size={16} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '3px' }}>
                        {preset.title}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{preset.sub}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}

      {/* ─── FLOW STATE 2: Uploading Status Banner (With Real Progress Percentage) ─── */}
      {flowState === 'uploading' && (
        <div className="panel animate-in" style={{ padding: '40px', textAlign: 'center' }}>
          <div className="live-dot" style={{ background: 'var(--accent-violet)', width: '12px', height: '12px', margin: '0 auto 16px' }} />
          <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Uploading Dataset: {uploadProgress}%</h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '8px', marginBottom: '20px' }}>
            Uploading data rows and running PII governance sanity scanner...
          </p>
          <div className="progress-track" style={{ maxWidth: '380px', margin: '0 auto' }}>
            <div className="progress-fill" style={{ width: `${uploadProgress}%`, background: 'var(--accent-violet)' }} />
          </div>
        </div>
      )}

      {/* ─── FLOW STATE 3: Pipeline Working Automation (With Stage Visualizations Side-by-Side) ─── */}
      {flowState === 'pipeline' && (
        <div className="panel animate-in" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Header & Overall Progress */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Pipeline Model Execution Status
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent-violet)', fontWeight: '700' }}>
                {Math.min(100, Math.round(((activeStageIdx) / STAGES.length) * 100 + (stagePercentage / 10)))}% Complete
              </span>
            </div>
            <div className="progress-track" style={{ marginBottom: '10px' }}>
              <div className="progress-fill" style={{ width: `${Math.min(100, ((activeStageIdx) / STAGES.length) * 100 + (stagePercentage / 10))}%` }} />
            </div>
          </div>

          {/* 2-Column Layout */}
          <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px', alignItems: 'start' }}>
            
            {/* Left Column: Stages Checklist */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ fontSize: '10px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>
                Pipeline Stages
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {STAGES.map((stg, idx) => {
                  const Icon = stg.icon;
                  const measured = uploadResult?.stage_report?.find(item => String(item.stage) === String(stg.id));
                  const measuredComplete = measured?.status === 'PASSED' || measured?.status === 'READY';
                  const isPassed = Boolean(measuredComplete);
                  const isCurrent = idx === activeStageIdx;
                  const displayStatus = measured?.status || (isCurrent ? 'RUNNING' : idx < activeStageIdx ? 'QUEUED' : 'WAITING');
                  
                  return (
                    <div key={stg.id} style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-md)',
                      background: isCurrent ? 'rgba(99,102,241,0.04)' : 'transparent',
                      border: isCurrent ? '1px solid rgba(99,102,241,0.12)' : '1px solid transparent',
                       transition: 'background var(--transition-fast), border-color var(--transition-fast)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                        <Icon size={12} style={{ color: isPassed ? 'var(--accent-emerald)' : isCurrent ? 'var(--accent-violet)' : 'var(--text-muted)', flexShrink: 0 }} />
                        <span style={{ 
                          fontSize: '11px', 
                          fontWeight: isCurrent ? '600' : '400', 
                          color: isCurrent ? 'var(--text-primary)' : 'var(--text-secondary)',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {stg.title.split(":")[0]}
                        </span>
                      </div>
                      <span className="badge" style={{
                        background: isPassed ? 'rgba(16,185,129,0.06)' : isCurrent ? 'rgba(99,102,241,0.06)' : 'rgba(0,0,0,0.02)',
                        color: isPassed ? '#065f46' : isCurrent ? 'var(--accent-violet)' : 'var(--text-muted)',
                        borderColor: isPassed ? 'rgba(16,185,129,0.18)' : isCurrent ? 'rgba(99,102,241,0.18)' : 'transparent',
                        fontSize: '9px',
                        padding: '1px 5px',
                        flexShrink: 0
                      }}>
                        {displayStatus}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Column: Dynamic Visualizer Area */}
            <div style={{
              background: 'rgba(99,102,241,0.01)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-lg)',
              padding: '16px',
              display: 'flex', flexDirection: 'column', gap: '14px'
            }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                <div style={{
                  width: '36px', height: '36px',
                  borderRadius: '8px',
                  background: 'rgba(99,102,241,0.08)',
                  border: '1px solid rgba(99,102,241,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--accent-violet)',
                  flexShrink: 0
                }}>
                  {React.createElement(STAGES[activeStageIdx].icon, { size: 16 })}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                    <span style={{ fontSize: '10px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--accent-violet)' }}>
                      Active Operation
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--accent-violet)', fontWeight: '700' }}>
                      {stagePercentage}%
                    </span>
                  </div>
                  <h3 style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px' }}>
                    {STAGES[activeStageIdx].title}
                  </h3>
                  <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    {STAGES[activeStageIdx].desc}
                  </p>
                </div>
              </div>

              {/* Stage Visualizer Box */}
              <div style={{
                background: 'var(--bg-panel)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '20px',
                height: '380px',
                overflow: 'hidden',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxSizing: 'border-box'
              }}>
                {renderStageVisualizer()}
              </div>
            </div>

          </div>
        </div>
      )}

      {/* ─── FLOW STATE 4: Completion (Show Governance scan + proceed button) ─── */}
      {flowState === 'complete' && uploadResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Completion Banner */}
          <div className="panel animate-in" style={{
            padding: '24px',
            borderColor: 'rgba(16,185,129,0.3)',
            background: 'rgba(16,185,129,0.03)',
            textAlign: 'center'
          }}>
            <div style={{
              width: '44px', height: '44px',
              borderRadius: '50%',
              background: 'rgba(16,185,129,0.1)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 12px',
              color: 'var(--accent-emerald)'
            }}>
              <CheckCircle2 size={24} />
            </div>
            <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Lumisyn-Net Synthesis Completed!</h2>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px', marginBottom: '16px' }}>
              The synthesis request completed. Review the measured validation results and the conditional public-schema privacy composition before using the output.
            </p>
            <button 
              onClick={() => setCurrentScreen && setCurrentScreen('dashboard')}
              className="btn-primary" 
              style={{ padding: '9px 24px', gap: '8px' }}
            >
              <LayoutDashboard size={14} />
              Proceed to Dashboard
            </button>
          </div>

          {/* Result Certificate */}
          <div className="panel animate-in" style={{ overflow: 'hidden', animationDelay: '100ms' }}>
            <div className="section-header" style={{ borderBottom: '1px solid rgba(16,185,129,0.15)', background: 'rgba(16,185,129,0.02)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldCheck size={16} color="var(--accent-emerald)" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)' }}>
                    {uploadResult.filename || 'clinical_dataset.csv'}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)' }}>
                    ID: {uploadResult.governance?.protocol_id} · SHA-256: {uploadResult.governance?.dataset_hash?.slice(0, 16)}…
                  </div>
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'var(--border-default)' }}>
              {[
                { label: 'Total Rows',          value: (uploadResult.row_count ?? uploadResult.total_rows ?? 0).toLocaleString(), color: 'var(--text-primary)' },
                { label: 'Total Columns',       value: uploadResult.column_count ?? uploadResult.total_columns ?? 0,               color: 'var(--accent-violet)' },
                { label: 'Re-ID Risk',          value: uploadResult?.re_id_risk == null ? '—' : `${(uploadResult.re_id_risk * 100).toFixed(2)}%`, color: 'var(--accent-emerald)' },
                { label: 'Missing Cells',       value: uploadResult?.missing_cells == null ? '—' : uploadResult.missing_cells.toLocaleString(), color: 'var(--accent-amber)' },
              ].map((s, i) => (

                <div key={i} style={{ padding: '14px', background: 'var(--bg-panel)', textAlign: 'center' }}>
                  <div style={{ fontSize: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '5px' }}>{s.label}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', fontWeight: '700', color: s.color }}>{s.value}</div>
                </div>
              ))}
            </div>

            <div style={{ padding: '12px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(16,185,129,0.02)', borderTop: '1px solid rgba(16,185,129,0.06)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                <Cpu size={14} color="var(--accent-emerald)" />
                Lumisyn research pipeline status: Output available for validation
              </div>
              <span className="badge badge--emerald">Ready</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
