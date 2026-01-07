/**
 * Governor Metrics Component
 * 
 * Displays Execution Governor state and metrics.
 */

import { useState, useEffect } from 'react';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('GovernorMetrics');

interface GovernorState {
  queueDepth: number;
  activeExecutions: number;
  maxConcurrentExecutions: number;
  registeredWorkers: number;
  healthyWorkers: number;
  totalMandatesProcessed: number;
  totalMandatesFailed: number;
  averageExecutionTime: number;
  currentBudgetUsage: number;
  globalBudgetLimit: number;
}

interface GovernorMetrics {
  totalMandatesProcessed: number;
  totalMandatesFailed: number;
  totalMandatesCompleted: number;
  averageExecutionTime: number;
  successRate: number;
  failureRate: number;
  byStatus: Record<string, number>;
  byWorker: Record<string, { processed: number; failed: number; avgTime: number }>;
}

interface GovernorMetricsProps {
  mandateId: string;
}

/**
 * Governor Metrics display component.
 */
export function GovernorMetrics({ mandateId }: GovernorMetricsProps) {
  const [state, setState] = useState<GovernorState | null>(null);
  const [metrics, setMetrics] = useState<GovernorMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const governorUrl = typeof window !== 'undefined' 
    ? (window as any).__GOVERNOR_URL__ || process.env.EXECUTION_GOVERNOR_URL || 'http://localhost:3000'
    : 'http://localhost:3000';

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch governor state
        const stateResponse = await fetch(`${governorUrl}/governor/state`, {
          signal: AbortSignal.timeout(3000) // 3 second timeout
        });
        if (stateResponse.ok) {
          const stateData = await stateResponse.json() as GovernorState;
          setState(stateData);
        } else {
          // Not available, but not an error - Governor is optional
          setLoading(false);
          return;
        }

        // Fetch governor metrics
        const metricsResponse = await fetch(`${governorUrl}/governor/metrics`, {
          signal: AbortSignal.timeout(3000) // 3 second timeout
        });
        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json() as GovernorMetrics;
          setMetrics(metricsData);
        }

        setLoading(false);
      } catch (err) {
        // Governor service is optional - don't treat as error
        if (err instanceof Error && (
          err.name === 'AbortError' || 
          err.message.includes('NetworkError') ||
          err.message.includes('Failed to fetch')
        )) {
          // Governor service not available - this is expected and OK
          logger.debug('Governor service not available (this is optional)');
          setLoading(false);
          setError(null); // Don't show error for optional service
        } else {
          logger.error('Error fetching governor data:', err);
          setError(err instanceof Error ? err.message : 'Unknown error');
          setLoading(false);
        }
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [governorUrl]);

  if (loading) {
    return (
      <div className="p-4">
        <div className="animate-pulse">Loading governor metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-yellow-500">
        <p>Error loading governor metrics: {error}</p>
        <p className="text-sm text-bolt-elements-textSecondary mt-2">
          Governor service may not be enabled or accessible at {governorUrl}
        </p>
      </div>
    );
  }

  // If Governor service is not available, show a friendly message instead of error
  if (!state && !loading) {
    return (
      <div className="p-4">
        <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6 text-center">
          <p className="text-bolt-elements-textSecondary mb-2">
            Execution Governor service is not available
          </p>
          <p className="text-sm text-bolt-elements-textTertiary">
            The Governor service provides queue management and execution metrics.
            It's optional and bolt.diy works without it.
          </p>
          <p className="text-xs text-bolt-elements-textTertiary mt-2">
            To enable: Start the Governor service at {governorUrl}
          </p>
        </div>
      </div>
    );
  }

  if (!state) {
    return null;
  }

  const budgetUsagePercent = state.globalBudgetLimit > 0
    ? (state.currentBudgetUsage / state.globalBudgetLimit) * 100
    : 0;

  const executionSlotUsage = state.maxConcurrentExecutions > 0
    ? (state.activeExecutions / state.maxConcurrentExecutions) * 100
    : 0;

  return (
    <div className="p-4 space-y-4">
      <h3 className="text-lg font-semibold mb-4">Execution Governor Metrics</h3>

      {/* Queue Status */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
          <div className="text-sm text-bolt-elements-textSecondary">Queue Depth</div>
          <div className="text-2xl font-bold text-bolt-elements-textPrimary">{state.queueDepth}</div>
        </div>

        <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
          <div className="text-sm text-bolt-elements-textSecondary">Active Executions</div>
          <div className="text-2xl font-bold text-bolt-elements-textPrimary">
            {state.activeExecutions} / {state.maxConcurrentExecutions}
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-accent-500 h-2 rounded-full transition-all"
                style={{ width: `${Math.min(executionSlotUsage, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Workers */}
      <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
        <div className="text-sm text-bolt-elements-textSecondary mb-2">Workers</div>
        <div className="flex items-center gap-4">
          <div>
            <span className="text-2xl font-bold text-green-500">{state.healthyWorkers}</span>
            <span className="text-sm text-bolt-elements-textSecondary ml-2">Healthy</span>
          </div>
          <div>
            <span className="text-2xl font-bold text-bolt-elements-textPrimary">{state.registeredWorkers}</span>
            <span className="text-sm text-bolt-elements-textSecondary ml-2">Total</span>
          </div>
        </div>
      </div>

      {/* Budget Usage */}
      <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
        <div className="text-sm text-bolt-elements-textSecondary mb-2">Budget Usage</div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-lg font-semibold">
            ${state.currentBudgetUsage.toFixed(2)} / ${state.globalBudgetLimit.toFixed(2)}
          </span>
          <span className="text-sm text-bolt-elements-textSecondary">
            {budgetUsagePercent.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${
              budgetUsagePercent > 90 ? 'bg-red-500' :
              budgetUsagePercent > 70 ? 'bg-yellow-500' :
              'bg-green-500'
            }`}
            style={{ width: `${Math.min(budgetUsagePercent, 100)}%` }}
          />
        </div>
      </div>

      {/* Statistics */}
      {metrics && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
            <div className="text-sm text-bolt-elements-textSecondary">Processed</div>
            <div className="text-xl font-bold text-bolt-elements-textPrimary">
              {metrics.totalMandatesProcessed}
            </div>
          </div>
          <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
            <div className="text-sm text-bolt-elements-textSecondary">Success Rate</div>
            <div className="text-xl font-bold text-green-500">
              {(metrics.successRate * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-bolt-elements-background-depth-2 p-4 rounded-lg">
            <div className="text-sm text-bolt-elements-textSecondary">Avg Time</div>
            <div className="text-xl font-bold text-bolt-elements-textPrimary">
              {Math.round(metrics.averageExecutionTime / 1000)}s
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

