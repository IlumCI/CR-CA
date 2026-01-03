import { useMemo } from 'react';
import type { ExecutionEvent } from '~/types/mandate';
import { classNames } from '~/utils/classNames';

interface DeploymentStatusProps {
  events: ExecutionEvent[];
}

/**
 * DeploymentStatus component displays deployment status and URLs.
 */
export function DeploymentStatus({ events }: DeploymentStatusProps) {
  const deploymentEvents = useMemo(() => {
    // Include both deployment_start, deployment_end, and deployment_status events
    return events.filter((event) => 
      event.type === 'deployment_status' || 
      event.type === 'deployment_start' || 
      event.type === 'deployment_end'
    );
  }, [events]);

  const latestDeployment = useMemo(() => {
    if (deploymentEvents.length === 0) return null;
    return deploymentEvents[deploymentEvents.length - 1];
  }, [deploymentEvents]);

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'complete':
        return 'text-green-500 bg-green-500/20';
      case 'running':
        return 'text-blue-500 bg-blue-500/20';
      case 'pending':
        return 'text-yellow-500 bg-yellow-500/20';
      case 'failed':
        return 'text-red-500 bg-red-500/20';
      default:
        return 'text-gray-500 bg-gray-500/20';
    }
  };

  if (deploymentEvents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">
          No deployment events yet. Deployment status will appear here when deployment is initiated.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Latest Deployment */}
      {latestDeployment && (
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-4">Latest Deployment</h3>

          <div className="space-y-4">
            {/* Status */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-bolt-elements-textSecondary">Status</span>
                <span
                  className={classNames(
                    'px-3 py-1 rounded text-sm font-medium',
                    getStatusColor(latestDeployment.data.deployment_status)
                  )}
                >
                  {(latestDeployment.data.deployment_status || 'unknown').toUpperCase()}
                </span>
              </div>
            </div>

            {/* Provider */}
            {latestDeployment.data.deployment_provider && (
              <div>
                <div className="text-sm text-bolt-elements-textSecondary mb-1">Provider</div>
                <div className="text-sm font-medium text-bolt-elements-textPrimary">
                  {latestDeployment.data.deployment_provider}
                </div>
              </div>
            )}

            {/* URL */}
            {latestDeployment.data.deployment_url && (
              <div>
                <div className="text-sm text-bolt-elements-textSecondary mb-1">Deployment URL</div>
                <a
                  href={latestDeployment.data.deployment_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-mono text-accent-500 hover:underline break-all"
                >
                  {latestDeployment.data.deployment_url}
                </a>
              </div>
            )}

            {/* Error */}
            {latestDeployment.data.error && (
              <div>
                <div className="text-sm text-bolt-elements-textSecondary mb-1">Error</div>
                <div className="text-sm text-red-500 bg-red-500/10 p-2 rounded">
                  {latestDeployment.data.error}
                </div>
              </div>
            )}

            {/* Timestamp */}
            <div>
              <div className="text-sm text-bolt-elements-textSecondary mb-1">Deployed At</div>
              <div className="text-sm text-bolt-elements-textPrimary">
                {new Date(latestDeployment.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Deployment History */}
      {deploymentEvents.length > 1 && (
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-4">Deployment History</h3>
          <div className="space-y-2">
            {deploymentEvents.map((event, index) => (
              <div
                key={`${event.timestamp}-${index}`}
                className="p-3 rounded border border-bolt-elements-borderColor bg-bolt-elements-background-depth-3"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-bolt-elements-textPrimary">
                    Iteration {event.iteration}
                  </span>
                  <span
                    className={classNames(
                      'px-2 py-1 rounded text-xs font-medium',
                      getStatusColor(event.data.deployment_status)
                    )}
                  >
                    {(event.data.deployment_status || 'unknown').toUpperCase()}
                  </span>
                </div>
                {event.data.deployment_url && (
                  <a
                    href={event.data.deployment_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-mono text-accent-500 hover:underline break-all"
                  >
                    {event.data.deployment_url}
                  </a>
                )}
                {event.data.error && (
                  <p className="text-xs text-red-500 mt-1">{event.data.error}</p>
                )}
                <div className="text-xs text-bolt-elements-textTertiary mt-1">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

