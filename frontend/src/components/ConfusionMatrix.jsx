import React, { useState } from 'react';

export default function ConfusionMatrix({ matrix, classes }) {
  const [hoveredCell, setHoveredCell] = useState(null);

  if (!matrix || !classes || matrix.length === 0) {
    return (
      <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
        No confusion matrix data available. Model has not been evaluated yet.
      </div>
    );
  }

  // Find max value in matrix for color intensity normalization
  let maxVal = 1;
  let totalSamples = 0;
  let correctSamples = 0;

  for (let r = 0; r < matrix.length; r++) {
    for (let c = 0; c < matrix[r].length; c++) {
      const val = matrix[r][c];
      if (val > maxVal) maxVal = val;
      totalSamples += val;
      if (r === c) correctSamples += val;
    }
  }

  const incorrectSamples = totalSamples - correctSamples;
  const accuracy = totalSamples > 0 ? ((correctSamples / totalSamples) * 100).toFixed(2) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Summary Stats Header */}
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div className="badge badge-purple">Total Evaluated: {totalSamples}</div>
        <div className="badge badge-emerald">Correct: {correctSamples} ({accuracy}%)</div>
        <div className="badge badge-amber">Incorrect: {incorrectSamples}</div>
      </div>

      {/* Interactive Matrix Grid */}
      <div style={{ overflowX: 'auto', paddingBottom: '8px' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: '4px', margin: '0 auto' }}>
          <thead>
            <tr>
              <th style={{ padding: '8px', fontSize: '0.72rem', color: 'var(--text-dim)', textAlign: 'right' }}>
                Actual \ Predicted
              </th>
              {classes.map((cls, idx) => (
                <th
                  key={idx}
                  title={cls}
                  style={{
                    padding: '8px 10px',
                    fontSize: '0.72rem',
                    color: 'var(--text-muted)',
                    fontWeight: 600,
                    maxWidth: '100px',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    textAlign: 'center'
                  }}
                >
                  {cls.split(' ')[0]}...
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => {
              const actualClass = classes[rIdx];
              return (
                <tr key={rIdx}>
                  <td
                    title={actualClass}
                    style={{
                      padding: '8px 12px',
                      fontSize: '0.72rem',
                      fontWeight: 600,
                      color: 'var(--text-muted)',
                      textAlign: 'right',
                      maxWidth: '140px',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}
                  >
                    {actualClass}
                  </td>
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const predClass = classes[cIdx];
                    const intensity = val > 0 ? Math.max(0.15, val / maxVal) : 0;
                    
                    const bgColor = isDiagonal
                      ? `rgba(16, 185, 129, ${intensity})`
                      : val > 0
                      ? `rgba(239, 68, 68, ${intensity * 0.85})`
                      : 'rgba(255, 255, 255, 0.02)';

                    const isHovered = hoveredCell && hoveredCell.r === rIdx && hoveredCell.c === cIdx;

                    return (
                      <td
                        key={cIdx}
                        onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, actual: actualClass, pred: predClass, count: val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{
                          width: '56px',
                          height: '48px',
                          textAlign: 'center',
                          backgroundColor: bgColor,
                          border: isHovered
                            ? '2px solid #8B5CF6'
                            : isDiagonal
                            ? '1px solid rgba(16, 185, 129, 0.4)'
                            : '1px solid var(--border-subtle)',
                          borderRadius: '6px',
                          fontSize: '0.85rem',
                          fontWeight: isDiagonal ? 700 : 500,
                          color: val > 0 ? '#FFFFFF' : 'var(--text-dim)',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease'
                        }}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Hover Info Tooltip Box */}
      <div style={{
        minHeight: '44px',
        padding: '10px 16px',
        borderRadius: '8px',
        backgroundColor: 'rgba(255, 255, 255, 0.03)',
        border: '1px solid var(--border-subtle)',
        fontSize: '0.8rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: hoveredCell ? '#FFFFFF' : 'var(--text-dim)'
      }}>
        {hoveredCell ? (
          <span>
            Actual: <strong style={{ color: '#A78BFA' }}>{hoveredCell.actual}</strong> &nbsp;➔&nbsp; 
            Predicted: <strong style={{ color: hoveredCell.actual === hoveredCell.pred ? '#10B981' : '#F87171' }}>{hoveredCell.pred}</strong> : &nbsp;
            <strong>{hoveredCell.count} samples</strong> ({((hoveredCell.count / totalSamples) * 100).toFixed(1)}% of test set)
          </span>
        ) : (
          <span>Hover over any matrix cell to inspect actual vs predicted class breakdowns.</span>
        )}
      </div>
    </div>
  );
}
