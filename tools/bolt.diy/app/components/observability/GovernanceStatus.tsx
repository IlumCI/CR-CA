import { useMemo } from 'react';
import type { ExecutionEvent } from '~/types/mandate';
import { classNames } from '~/utils/classNames';

interface GovernanceStatusProps {
  events: ExecutionEvent[];
}

/**
 * GovernanceStatus component displays ESG scores, risk assessments, and compliance status.
 */
export function GovernanceStatus({ events }: GovernanceStatusProps) {
  const governanceEvents = useMemo(() => {
    return events.filter((event) => event.type === 'governance_check');
  }, [events]);

  const latestGovernance = useMemo(() => {
    if (governanceEvents.length === 0) return null;
    return governanceEvents[governanceEvents.length - 1];
  }, [governanceEvents]);

  const getRiskColor = (level?: string) => {
    switch (level) {
      case 'critical':
        return 'text-red-600 bg-red-500/20';
      case 'high':
        return 'text-orange-600 bg-orange-500/20';
      case 'medium':
        return 'text-yellow-600 bg-yellow-500/20';
      case 'low':
        return 'text-green-600 bg-green-500/20';
      default:
        return 'text-gray-600 bg-gray-500/20';
    }
  };

  if (governanceEvents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">
          No governance checks yet. Governance status will appear here during execution.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Latest Governance Status */}
      {latestGovernance && (
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-4">
            Latest Governance Check (Iteration {latestGovernance.iteration})
          </h3>

          {/* ESG Scores */}
          {latestGovernance.data.esg_score && (
            <div className="mb-6">
              <h4 className="text-md font-medium text-bolt-elements-textPrimary mb-3">ESG Scores</h4>
              <div className="grid grid-cols-2 gap-4">
                {typeof latestGovernance.data.esg_score === 'object' ? (
                  <>
                    <div>
                      <div className="text-sm text-bolt-elements-textSecondary mb-1">Environmental</div>
                      <div className="text-2xl font-bold text-green-500">
                        {(latestGovernance.data.esg_score as any).environmental_score?.toFixed(1) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-bolt-elements-textSecondary mb-1">Social</div>
                      <div className="text-2xl font-bold text-blue-500">
                        {(latestGovernance.data.esg_score as any).social_score?.toFixed(1) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-bolt-elements-textSecondary mb-1">Governance</div>
                      <div className="text-2xl font-bold text-purple-500">
                        {(latestGovernance.data.esg_score as any).governance_score?.toFixed(1) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-bolt-elements-textSecondary mb-1">Overall</div>
                      <div className="text-2xl font-bold text-accent-500">
                        {(latestGovernance.data.esg_score as any).overall_score?.toFixed(1) || 'N/A'}
                      </div>
                    </div>
                  </>
                ) : (
                  <div>
                    <div className="text-sm text-bolt-elements-textSecondary mb-1">Overall Score</div>
                    <div className="text-2xl font-bold text-accent-500">
                      {Number(latestGovernance.data.esg_score).toFixed(1)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Risk Assessment */}
          {latestGovernance.data.risk_assessment && (
            <div className="mb-6">
              <h4 className="text-md font-medium text-bolt-elements-textPrimary mb-3">Risk Assessment</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-bolt-elements-textSecondary">Risk Level</span>
                    <span
                      className={classNames(
                        'px-3 py-1 rounded text-sm font-medium',
                        getRiskColor(
                          (latestGovernance.data.risk_assessment as any)?.risk_level || 'unknown'
                        )
                      )}
                    >
                      {(latestGovernance.data.risk_assessment as any)?.risk_level?.toUpperCase() || 'UNKNOWN'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-bolt-elements-textSecondary">Risk Score</span>
                    <span className="text-sm font-medium text-bolt-elements-textPrimary">
                      {(latestGovernance.data.risk_assessment as any)?.risk_score?.toFixed(1) || 'N/A'} / 100
                    </span>
                  </div>
                </div>
                {(latestGovernance.data.risk_assessment as any)?.mitigation_strategy && (
                  <div>
                    <div className="text-sm text-bolt-elements-textSecondary mb-1">Mitigation Strategy</div>
                    <p className="text-sm text-bolt-elements-textPrimary">
                      {(latestGovernance.data.risk_assessment as any).mitigation_strategy}
                    </p>
                  </div>
                )}
                {(latestGovernance.data.risk_assessment as any)?.impact_areas &&
                  (latestGovernance.data.risk_assessment as any).impact_areas.length > 0 && (
                    <div>
                      <div className="text-sm text-bolt-elements-textSecondary mb-1">Impact Areas</div>
                      <div className="flex flex-wrap gap-2">
                        {((latestGovernance.data.risk_assessment as any).impact_areas as string[]).map(
                          (area, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-1 rounded text-xs bg-bolt-elements-background-depth-3 text-bolt-elements-textSecondary"
                            >
                              {area}
                            </span>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* Budget Consumption */}
          {latestGovernance.data.budget_consumed !== undefined && (
            <div>
              <h4 className="text-md font-medium text-bolt-elements-textPrimary mb-3">Budget Consumption</h4>
              <div className="text-2xl font-bold text-bolt-elements-textPrimary">
                ${Number(latestGovernance.data.budget_consumed).toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Governance History */}
      {governanceEvents.length > 1 && (
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-4">Governance History</h3>
          <div className="space-y-2">
            {governanceEvents.map((event, index) => (
              <div
                key={`${event.timestamp}-${index}`}
                className="p-3 rounded border border-bolt-elements-borderColor bg-bolt-elements-background-depth-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-bolt-elements-textPrimary">
                    Iteration {event.iteration}
                  </span>
                  <span className="text-xs text-bolt-elements-textTertiary">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                {event.data.message && (
                  <p className="text-sm text-bolt-elements-textSecondary mt-1">{event.data.message}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

