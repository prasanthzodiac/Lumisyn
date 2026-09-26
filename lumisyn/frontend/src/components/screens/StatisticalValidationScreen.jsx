import React from 'react';
import StatisticalComparisonReport from '../StatisticalComparisonReport';

export default function StatisticalValidationScreen({ sessionData }) {
  return (
    <div className="screen-enter">
      <StatisticalComparisonReport sessionData={sessionData} />
    </div>
  );
}
