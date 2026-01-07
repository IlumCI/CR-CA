/**
 * Workflow Timeline Component.
 * 
 * Visual timeline showing all phases of the workflow from proposal to deployment.
 */

import { useState } from 'react';
import { WorkflowPhase } from './WorkflowPhase';
import type { WorkflowStatus as WorkflowStatusType } from '~/routes/api.workflow-status.$proposalId';
import { classNames } from '~/utils/classNames';

interface WorkflowTimelineProps {
  workflowStatus: WorkflowStatusType;
}

export function WorkflowTimeline({ workflowStatus }: WorkflowTimelineProps) {
  const [expandedPhase, setExpandedPhase] = useState<string | null>(null);

  const statusColors = {
    pending: "text-bolt-elements-textTertiary",
    running: "text-blue-500",
    success: "text-green-500",
    failed: "text-red-500",
    skipped: "text-yellow-500"
  };

  const statusIcons = {
    pending: "○",
    running: "⟳",
    success: "✓",
    failed: "✗",
    skipped: "⊘"
  };

  return (
    <div className="space-y-6">
      {/* Overall Status */}
      <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-bolt-elements-textPrimary">
              Overall Status
            </h2>
            <p className="text-sm text-bolt-elements-textSecondary mt-1">
              {workflowStatus.phases.length} phases • Current: {workflowStatus.current_phase}
            </p>
          </div>
          <div className={classNames(
            "px-4 py-2 rounded-lg font-medium",
            statusColors[workflowStatus.overall_status] || statusColors.pending,
            "bg-bolt-elements-background-depth-1"
          )}>
            {statusIcons[workflowStatus.overall_status] || "○"} {workflowStatus.overall_status.toUpperCase()}
          </div>
        </div>

        {workflowStatus.deployment_url && (
          <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
            <p className="text-sm text-green-500 font-medium mb-1">Deployment URL</p>
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

        {workflowStatus.test_results && (
          <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <p className="text-sm text-blue-500 font-medium mb-1">Test Results</p>
            <div className="text-sm text-bolt-elements-textSecondary">
              Status: {workflowStatus.test_results.status} • 
              {workflowStatus.test_results.tests_passed ? " ✓ Passed" : " ✗ Failed"} • 
              Coverage: {workflowStatus.test_results.coverage ? `${(workflowStatus.test_results.coverage * 100).toFixed(1)}%` : "N/A"}
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6">
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-6">
          Workflow Phases
        </h3>
        
        <div className="space-y-4">
          {workflowStatus.phases.map((phase, index) => (
            <div key={phase.id} className="relative">
              {/* Timeline connector */}
              {index < workflowStatus.phases.length - 1 && (
                <div className="absolute left-6 top-12 w-0.5 h-full bg-bolt-elements-borderColor" />
              )}
              
              <div className="flex items-start gap-4">
                {/* Status indicator */}
                <div className={classNames(
                  "flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg",
                  phase.status === "success" ? "bg-green-500/20 text-green-500" :
                  phase.status === "failed" ? "bg-red-500/20 text-red-500" :
                  phase.status === "running" ? "bg-blue-500/20 text-blue-500 animate-pulse" :
                  "bg-bolt-elements-background-depth-1 text-bolt-elements-textTertiary"
                )}>
                  {statusIcons[phase.status] || "○"}
                </div>

                {/* Phase content */}
                <div className="flex-1">
                  <WorkflowPhase
                    phase={phase}
                    isExpanded={expandedPhase === phase.id}
                    onToggle={() => setExpandedPhase(expandedPhase === phase.id ? null : phase.id)}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-4">
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Total Phases</p>
          <p className="text-2xl font-bold text-bolt-elements-textPrimary">
            {workflowStatus.phases.length}
          </p>
        </div>
        <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-4">
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Completed</p>
          <p className="text-2xl font-bold text-green-500">
            {workflowStatus.phases.filter(p => p.status === "success").length}
          </p>
        </div>
        <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-4">
          <p className="text-sm text-bolt-elements-textSecondary mb-1">Duration</p>
          <p className="text-2xl font-bold text-bolt-elements-textPrimary">
            {workflowStatus.phases.length > 0 && workflowStatus.phases[0].timestamp && 
             workflowStatus.phases[workflowStatus.phases.length - 1].timestamp
              ? `${Math.round((workflowStatus.phases[workflowStatus.phases.length - 1].timestamp! - workflowStatus.phases[0].timestamp!) / 60)}m`
              : "N/A"}
          </p>
        </div>
      </div>
    </div>
  );
}

