import { useState, useEffect } from 'react';
import type { ExecutionEvent } from '~/types/mandate';
import type { Mandate } from '~/types/mandate';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('MandateDetails');

interface MandateDetailsProps {
  mandateId: string;
  events: ExecutionEvent[];
}

/**
 * MandateDetails component displays mandate configuration and metadata.
 */
export function MandateDetails({ mandateId, events }: MandateDetailsProps) {
  const [mandate, setMandate] = useState<Mandate | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Try to extract mandate from events (if it was included)
    const mandateEvent = events.find((e) => e.type === 'iteration_start' && e.data.mandate);
    if (mandateEvent?.data.mandate) {
      setMandate(mandateEvent.data.mandate as Mandate);
      setLoading(false);
      return;
    }

    // Otherwise, try to fetch from API (if available)
    const fetchMandate = async () => {
      try {
        // This would require a new API endpoint to fetch mandate details
        // For now, we'll just show what we can from events
        setLoading(false);
      } catch (err) {
        logger.error('Error fetching mandate:', err);
        setLoading(false);
      }
    };

    fetchMandate();
  }, [mandateId, events]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">Loading mandate details...</p>
      </div>
    );
  }

  if (!mandate) {
    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-2">Mandate ID</h3>
          <p className="text-sm font-mono text-bolt-elements-textSecondary bg-bolt-elements-background-depth-3 p-2 rounded">
            {mandateId}
          </p>
        </div>
        <div>
          <p className="text-bolt-elements-textSecondary">
            Mandate details will be available once execution starts. Check the timeline for execution events.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Basic Info */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Mandate Information</h3>
        <div className="space-y-2">
          <div>
            <span className="text-sm font-medium text-bolt-elements-textSecondary">ID:</span>
            <span className="ml-2 text-sm font-mono text-bolt-elements-textPrimary">{mandate.mandate_id}</span>
          </div>
        </div>
      </div>

      {/* Objectives */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Objectives</h3>
        <ul className="list-disc list-inside space-y-1 text-sm text-bolt-elements-textSecondary">
          {mandate.objectives.map((obj, idx) => (
            <li key={idx}>{obj}</li>
          ))}
        </ul>
      </div>

      {/* Constraints */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Constraints</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-bolt-elements-textSecondary">Language:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">{mandate.constraints.language}</span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Max Dependencies:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">{mandate.constraints.maxDependencies}</span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">No Network:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">
              {mandate.constraints.noNetwork ? 'Yes' : 'No'}
            </span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Max Files:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">{mandate.constraints.maxFiles}</span>
          </div>
        </div>
      </div>

      {/* Budget */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Budget</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-bolt-elements-textSecondary">Tokens:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">
              {mandate.budget.token.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Time:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">{mandate.budget.time}s</span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Cost:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">${mandate.budget.cost.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Deliverables */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Deliverables</h3>
        <ul className="list-disc list-inside space-y-1 text-sm text-bolt-elements-textSecondary">
          {mandate.deliverables.map((deliverable, idx) => (
            <li key={idx}>{deliverable}</li>
          ))}
        </ul>
      </div>

      {/* Iteration Config */}
      <div>
        <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Iteration Configuration</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-bolt-elements-textSecondary">Max Iterations:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">
              {mandate.iteration_config.max_iterations}
            </span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Test Required:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">
              {mandate.iteration_config.test_required ? 'Yes' : 'No'}
            </span>
          </div>
          <div>
            <span className="text-bolt-elements-textSecondary">Quality Threshold:</span>
            <span className="ml-2 text-bolt-elements-textPrimary">
              {mandate.iteration_config.quality_threshold}
            </span>
          </div>
        </div>
      </div>

      {/* Deployment Config */}
      {mandate.deployment && (
        <div>
          <h3 className="text-lg font-semibold text-bolt-elements-textPrimary mb-3">Deployment</h3>
          <div className="text-sm space-y-1">
            <div>
              <span className="text-bolt-elements-textSecondary">Provider:</span>
              <span className="ml-2 text-bolt-elements-textPrimary">{mandate.deployment.provider}</span>
            </div>
            <div>
              <span className="text-bolt-elements-textSecondary">Auto Deploy:</span>
              <span className="ml-2 text-bolt-elements-textPrimary">
                {mandate.deployment.auto_deploy ? 'Yes' : 'No'}
              </span>
            </div>
            {mandate.deployment.target && (
              <div>
                <span className="text-bolt-elements-textSecondary">Target:</span>
                <span className="ml-2 text-bolt-elements-textPrimary">{mandate.deployment.target}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

