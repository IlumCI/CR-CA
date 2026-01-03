/**
 * Execution event system for mandate tracking and observability.
 * 
 * Provides structured event emission and collection for autonomous
 * code generation cycles with governance oversight.
 */

import type {
  ExecutionEvent,
  ExecutionEventType,
  ExecutionEventData,
  ESGScore,
  RiskAssessment,
} from '~/types/mandate';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('execution-events');

/**
 * Event emitter for mandate execution.
 * Collects and manages execution events for observability and governance reporting.
 */
export class ExecutionEventEmitter {
  private events: ExecutionEvent[] = [];
  private listeners: Map<string, Set<(event: ExecutionEvent) => void>> = new Map();
  private mandateId: string;
  private currentIteration: number = 0;

  constructor(mandateId: string) {
    this.mandateId = mandateId;
  }

  /**
   * Emit an execution event.
   */
  emit(
    type: ExecutionEventType,
    data: ExecutionEventData,
    metadata?: ExecutionEvent['metadata']
  ): void {
    const event: ExecutionEvent = {
      mandate_id: this.mandateId,
      iteration: this.currentIteration,
      type,
      timestamp: Date.now(),
      data,
      metadata: metadata || {},
    };

    this.events.push(event);

    // Notify listeners
    const typeListeners = this.listeners.get(type);
    if (typeListeners) {
      typeListeners.forEach((listener) => {
        try {
          listener(event);
        } catch (error) {
          logger.error(`Error in event listener for ${type}:`, error);
        }
      });
    }

    // Notify all listeners
    const allListeners = this.listeners.get('*');
    if (allListeners) {
      allListeners.forEach((listener) => {
        try {
          listener(event);
        } catch (error) {
          logger.error('Error in wildcard event listener:', error);
        }
      });
    }

    logger.debug(`Emitted event: ${type} for mandate ${this.mandateId}, iteration ${this.currentIteration}`);
  }

  /**
   * Subscribe to events of a specific type.
   */
  on(type: ExecutionEventType | '*', listener: (event: ExecutionEvent) => void): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(listener);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(type);
      if (listeners) {
        listeners.delete(listener);
      }
    };
  }

  /**
   * Set current iteration number.
   */
  setIteration(iteration: number): void {
    this.currentIteration = iteration;
  }

  /**
   * Get all events.
   */
  getEvents(): ExecutionEvent[] {
    return [...this.events];
  }

  /**
   * Get events by type.
   */
  getEventsByType(type: ExecutionEventType): ExecutionEvent[] {
    return this.events.filter((event) => event.type === type);
  }

  /**
   * Get events for current iteration.
   */
  getIterationEvents(iteration: number): ExecutionEvent[] {
    return this.events.filter((event) => event.iteration === iteration);
  }

  /**
   * Get latest event of a specific type.
   */
  getLatestEvent(type: ExecutionEventType): ExecutionEvent | undefined {
    const events = this.getEventsByType(type);
    return events.length > 0 ? events[events.length - 1] : undefined;
  }

  /**
   * Clear all events (use with caution).
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Helper methods for common event types.
   */
  emitIterationStart(iteration: number, metadata?: ExecutionEvent['metadata']): void {
    this.setIteration(iteration);
    this.emit(
      'iteration_start',
      {
        status: 'running',
        iteration_number: iteration,
      },
      metadata
    );
  }

  emitIterationEnd(
    iteration: number,
    status: 'success' | 'failed' | 'stopped',
    metadata?: ExecutionEvent['metadata']
  ): void {
    this.emit(
      'iteration_end',
      {
        status,
        iteration_number: iteration,
      },
      metadata
    );
  }

  emitLog(level: 'info' | 'warn' | 'error' | 'debug', message: string, source?: string): void {
    this.emit('log', {
      level,
      message,
      source,
    });
  }

  emitError(message: string, source?: string, metadata?: ExecutionEvent['metadata']): void {
    this.emit(
      'error',
      {
        level: 'error',
        message,
        source,
      },
      metadata
    );
  }

  emitDiff(filesChanged: string[], linesAdded?: number, linesRemoved?: number): void {
    this.emit('diff', {
      files_changed: filesChanged,
      lines_added: linesAdded,
      lines_removed: linesRemoved,
    });
  }

  emitGovernanceCheck(
    esgScore?: ESGScore,
    riskAssessment?: RiskAssessment,
    budgetConsumed?: ExecutionEventData['budget_consumed'],
    complianceStatus?: Record<string, boolean>
  ): void {
    this.emit('governance_check', {
      esg_score: esgScore,
      risk_assessment: riskAssessment,
      budget_consumed: budgetConsumed,
      compliance_status: complianceStatus,
    });
  }

  emitDeploymentStart(provider: string): void {
    this.emit('deployment_status', {
      deployment_provider: provider,
      deployment_status: 'running',
      message: `Deployment to ${provider} started`,
    });
  }

  emitDeploymentEnd(
    provider: string,
    status: 'success' | 'failed',
    url?: string,
    error?: string
  ): void {
    this.emit('deployment_status', {
      deployment_provider: provider,
      deployment_status: status === 'success' ? 'complete' : 'failed',
      deployment_url: url,
      error: error,
      message: status === 'success' 
        ? `Deployment to ${provider} completed successfully` 
        : `Deployment to ${provider} failed: ${error || 'Unknown error'}`,
    });
  }

  emitBudgetWarning(
    budgetConsumed: ExecutionEventData['budget_consumed'],
    threshold: number = 0.8
  ): void {
    if (budgetConsumed) {
      const { tokens, cost, time } = budgetConsumed;
      // Check if any budget category exceeds threshold
      // This is a simplified check - actual implementation would compare against mandate budget
      this.emit('budget_warning', {
        budget_consumed: budgetConsumed,
        message: `Budget consumption approaching limit (${(threshold * 100).toFixed(0)}%)`,
      });
    }
  }

  emitConstraintViolation(violationType: string, details: string): void {
    this.emit('constraint_violation', {
      violation_type: violationType,
      violation_details: details,
      level: 'error',
    });
  }
}

/**
 * Global event emitter registry for managing multiple mandate executions.
 */
class ExecutionEventRegistry {
  private emitters: Map<string, ExecutionEventEmitter> = new Map();

  /**
   * Get or create event emitter for a mandate.
   */
  getEmitter(mandateId: string): ExecutionEventEmitter {
    if (!this.emitters.has(mandateId)) {
      this.emitters.set(mandateId, new ExecutionEventEmitter(mandateId));
    }
    return this.emitters.get(mandateId)!;
  }

  /**
   * Remove emitter for a mandate (cleanup after completion).
   */
  removeEmitter(mandateId: string): void {
    this.emitters.delete(mandateId);
  }

  /**
   * Get all active mandate IDs.
   */
  getActiveMandates(): string[] {
    return Array.from(this.emitters.keys());
  }

  /**
   * Clear all emitters (use with caution).
   */
  clear(): void {
    this.emitters.clear();
  }
}

export const eventRegistry = new ExecutionEventRegistry();

