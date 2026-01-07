/**
 * Workflow Status Component.
 * 
 * Displays overall workflow status and key metrics.
 */

import type { WorkflowStatus as WorkflowStatusType } from '~/routes/api.workflow-status.$proposalId';

interface WorkflowStatusProps {
  workflowStatus: WorkflowStatusType;
}

export function WorkflowStatus({ workflowStatus }: WorkflowStatusProps) {
  const successCount = workflowStatus.phases.filter(p => p.status === "success").length;
  const failedCount = workflowStatus.phases.filter(p => p.status === "failed").length;
  const runningCount = workflowStatus.phases.filter(p => p.status === "running").length;
  const pendingCount = workflowStatus.phases.filter(p => p.status === "pending").length;

  return (
    <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6">
      <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-4">
        Status Summary
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Success</p>
          <p className="text-2xl font-bold text-green-500">{successCount}</p>
        </div>
        <div>
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Failed</p>
          <p className="text-2xl font-bold text-red-500">{failedCount}</p>
        </div>
        <div>
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Running</p>
          <p className="text-2xl font-bold text-blue-500">{runningCount}</p>
        </div>
        <div>
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Pending</p>
          <p className="text-2xl font-bold text-bolt-elements-textTertiary">{pendingCount}</p>
        </div>
      </div>

      {workflowStatus.deployment_url && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <p className="text-sm text-green-500 font-medium mb-1">Deployment</p>
          <a 
            href={workflowStatus.deployment_url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-sm text-green-400 hover:text-green-300 underline"
          >
            {workflowStatus.deployment_url}
          </a>
        </div>
      )}
    </div>
  );
}

