/**
 * Workflow Phase Component.
 * 
 * Individual phase display with expandable details.
 */

import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import type { WorkflowPhase as WorkflowPhaseType } from '~/routes/api.workflow-status.$proposalId';
import { classNames } from '~/utils/classNames';

interface WorkflowPhaseProps {
  phase: WorkflowPhaseType;
  isExpanded: boolean;
  onToggle: () => void;
}

export function WorkflowPhase({ phase, isExpanded, onToggle }: WorkflowPhaseProps) {
  const statusColors = {
    pending: "text-bolt-elements-textTertiary",
    running: "text-blue-500",
    success: "text-green-500",
    failed: "text-red-500",
    skipped: "text-yellow-500"
  };

  const formatTimestamp = (timestamp?: number) => {
    if (!timestamp) return "N/A";
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return "N/A";
    if (duration < 60) return `${Math.round(duration)}s`;
    return `${Math.round(duration / 60)}m ${Math.round(duration % 60)}s`;
  };

  return (
    <div className="bg-bolt-elements-background-depth-1 border border-bolt-elements-borderColor rounded-lg">
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-bolt-elements-background-depth-2 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 text-left">
          {isExpanded ? (
            <ChevronDownIcon className="w-5 h-5 text-bolt-elements-textSecondary" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-bolt-elements-textSecondary" />
          )}
          <div>
            <h4 className="font-semibold text-bolt-elements-textPrimary">
              {phase.name}
            </h4>
            <div className="flex items-center gap-4 mt-1">
              <span className={classNames(
                "text-xs font-medium",
                statusColors[phase.status] || statusColors.pending
              )}>
                {phase.status.toUpperCase()}
              </span>
              {phase.timestamp && (
                <span className="text-xs text-bolt-elements-textSecondary">
                  {formatTimestamp(phase.timestamp)}
                </span>
              )}
              {phase.duration && (
                <span className="text-xs text-bolt-elements-textSecondary">
                  Duration: {formatDuration(phase.duration)}
                </span>
              )}
            </div>
          </div>
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-bolt-elements-borderColor">
          {phase.error && (
            <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-500 font-medium mb-1">Error</p>
              <p className="text-sm text-red-400">{phase.error}</p>
            </div>
          )}

          {phase.details && (
            <div className="mt-3 space-y-2">
              {Object.entries(phase.details).map(([key, value]) => (
                <div key={key} className="text-sm">
                  <span className="text-bolt-elements-textSecondary font-medium">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                  </span>
                  <span className="ml-2 text-bolt-elements-textPrimary">
                    {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {!phase.details && !phase.error && (
            <div className="mt-3 text-sm text-bolt-elements-textSecondary">
              No additional details available
            </div>
          )}
        </div>
      )}
    </div>
  );
}

