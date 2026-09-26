import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import TopHeader from './components/TopHeader';
import { apiUrl } from './api/client';

import DashboardScreen from './components/screens/DashboardScreen';
import DatasetUploadScreen from './components/DatasetUploader';
import DatasetIntelligenceScreen from './components/screens/DatasetIntelligenceScreen';
import DataVisualizationScreen from './components/screens/DataVisualizationScreen';
import ClinicalKnowledgeGraphScreen from './components/screens/ClinicalKnowledgeGraphScreen';
import PipelineOverviewScreen from './components/screens/PipelineOverviewScreen';
import StageDetailsHVAEScreen from './components/screens/StageDetailsHVAEScreen';
import AdaptivePrivacyControllerScreen from './components/screens/AdaptivePrivacyControllerScreen';
import FlowMatchingAnimationScreen from './components/screens/FlowMatchingAnimationScreen';
import OriginalVsSyntheticScreen from './components/screens/OriginalVsSyntheticScreen';
import StatisticalValidationScreen from './components/screens/StatisticalValidationScreen';
import ClinicalConsistencyScreen from './components/screens/ClinicalConsistencyScreen';
import ExperimentHistoryScreen from './components/screens/ExperimentHistoryScreen';
import ReportsAndDownloadsScreen from './components/screens/ReportsAndDownloadsScreen';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('upload');
  const [sessionData, setSessionData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [generationResult, setGenerationResult] = useState(null);

  const handleGenerationComplete = async (_res) => {
    if (!sessionData) return;
    setGenerationResult(_res);
    setSessionData(prev => prev ? { ...prev, lastRunAt: Date.now() } : prev);
    try {
      const cmpRes = await fetch(apiUrl(`/api/compare/${sessionData.session_id}`));
      const cmpData = await cmpRes.json();
      setComparisonData(cmpData);
    } catch (err) {
      console.error("Comparison fetch error:", err);
    }
  };

  const screenTitles = {
    dashboard: "01. Dashboard Overview",
    upload: "02. Upload Dataset & Governance",
    intelligence: "03. Dataset Intelligence Engine (UDIE)",
    datavis: "04. Exploratory Data Visualization",
    graph: "05. Multi-Relational Clinical Knowledge Graph",
    pipeline: "06. End-to-End Pipeline Workflow Overview",
    stage_hvae: "07. Stage Details – HVAE Representation",
    privacy: "08. Adaptive Privacy Controller (APC v5)",
    flow_anim: "09. Flow Matching Trajectory Particle Simulator",
    comparison: "10. Original vs Synthetic Empirical Distribution Comparison",
    validation: "11. Statistical Validation & Hypothesis Testing",
    consistency: "12. Clinical Relationship Consistency Matrix",
    history: "13. Experiment History & Audit Trail",
    reports: "14. Reports Generator & Artifact Downloads"
  };

  const renderScreen = () => {
    switch (currentScreen) {
      case 'dashboard':
        return <DashboardScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'upload':
        return <DatasetUploadScreen onDatasetLoaded={(d, cmp = null) => { setSessionData(d); setComparisonData(cmp); setGenerationResult(d.generationResult || null); }} setCurrentScreen={setCurrentScreen} />;
      case 'intelligence':
        return <DatasetIntelligenceScreen sessionData={sessionData} />;
      case 'datavis':
        return <DataVisualizationScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'graph':
        return <ClinicalKnowledgeGraphScreen sessionData={sessionData} />;
      case 'pipeline':
        return <PipelineOverviewScreen sessionData={sessionData} generationResult={generationResult} onGenerationComplete={handleGenerationComplete} />;
      case 'stage_hvae':
        return <StageDetailsHVAEScreen sessionData={sessionData} generationResult={generationResult || sessionData?.generationResult} />;
      case 'privacy':
        return <AdaptivePrivacyControllerScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'flow_anim':
        return <FlowMatchingAnimationScreen sessionData={sessionData} />;
      case 'comparison':
        return <OriginalVsSyntheticScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'validation':
        return <StatisticalValidationScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'consistency':
        return <ClinicalConsistencyScreen sessionData={sessionData} comparisonData={comparisonData} />;
      case 'history':
        return <ExperimentHistoryScreen sessionData={sessionData} />;
      case 'reports':
        return <ReportsAndDownloadsScreen sessionData={sessionData} />;
      default:
        return <DashboardScreen />;
    }
  };

  return (
    <div className="app-shell">
      {/* Persistent Left Sidebar Navigation */}
      {currentScreen !== 'upload' && (
        <Sidebar currentScreen={currentScreen} setCurrentScreen={setCurrentScreen} sessionData={sessionData} />
      )}

      {/* Main Content Area */}
      <div className="main-area">
        {currentScreen !== 'upload' && (
          <TopHeader title={screenTitles[currentScreen] || "Dashboard"} sessionData={sessionData} />
        )}

        <main className="screen-container" style={currentScreen === 'upload' ? { display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' } : {}}>
          {renderScreen()}
        </main>
      </div>
    </div>
  );
}
