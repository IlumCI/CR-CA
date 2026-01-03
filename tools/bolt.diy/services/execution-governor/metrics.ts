/**
 * Metrics collection for Execution Governor
 * 
 * Tracks execution statistics and performance metrics.
 */

import type { GovernorState, ExecutionStatus } from './types';

export interface ExecutionMetrics {
  totalMandatesProcessed: number;
  totalMandatesFailed: number;
  totalMandatesCompleted: number;
  averageExecutionTime: number;
  executionTimes: number[];
  successRate: number;
  failureRate: number;
  byStatus: Record<string, number>;
  byWorker: Record<string, { processed: number; failed: number; avgTime: number }>;
  recentExecutions: Array<{
    mandateId: string;
    status: string;
    duration?: number;
    timestamp: number;
  }>;
}

export class MetricsCollector {
  private metrics: ExecutionMetrics;
  private maxRecentExecutions: number = 100;

  constructor() {
    this.metrics = {
      totalMandatesProcessed: 0,
      totalMandatesFailed: 0,
      totalMandatesCompleted: 0,
      averageExecutionTime: 0,
      executionTimes: [],
      successRate: 0,
      failureRate: 0,
      byStatus: {},
      byWorker: {},
      recentExecutions: [],
    };
  }

  /**
   * Record mandate execution start.
   */
  recordStart(mandateId: string, workerId: string): void {
    this.metrics.totalMandatesProcessed++;
    this.metrics.byStatus['executing'] = (this.metrics.byStatus['executing'] || 0) + 1;
    
    if (!this.metrics.byWorker[workerId]) {
      this.metrics.byWorker[workerId] = { processed: 0, failed: 0, avgTime: 0 };
    }
    this.metrics.byWorker[workerId].processed++;
  }

  /**
   * Record mandate execution completion.
   */
  recordCompletion(mandateId: string, workerId: string, duration: number, success: boolean): void {
    if (success) {
      this.metrics.totalMandatesCompleted++;
      this.metrics.byStatus['completed'] = (this.metrics.byStatus['completed'] || 0) + 1;
    } else {
      this.metrics.totalMandatesFailed++;
      this.metrics.byStatus['failed'] = (this.metrics.byStatus['failed'] || 0) + 1;
      if (this.metrics.byWorker[workerId]) {
        this.metrics.byWorker[workerId].failed++;
      }
    }

    // Update execution times
    this.metrics.executionTimes.push(duration);
    if (this.metrics.executionTimes.length > 1000) {
      // Keep only last 1000 for average calculation
      this.metrics.executionTimes = this.metrics.executionTimes.slice(-1000);
    }

    // Calculate average
    const sum = this.metrics.executionTimes.reduce((a, b) => a + b, 0);
    this.metrics.averageExecutionTime = sum / this.metrics.executionTimes.length;

    // Update success/failure rates
    const total = this.metrics.totalMandatesProcessed;
    this.metrics.successRate = total > 0 ? this.metrics.totalMandatesCompleted / total : 0;
    this.metrics.failureRate = total > 0 ? this.metrics.totalMandatesFailed / total : 0;

    // Update worker average time
    if (this.metrics.byWorker[workerId]) {
      const worker = this.metrics.byWorker[workerId];
      const workerTimes = this.metrics.executionTimes.filter((_, i) => {
        // Simplified: assume recent executions are from this worker
        return i >= this.metrics.executionTimes.length - worker.processed;
      });
      if (workerTimes.length > 0) {
        worker.avgTime = workerTimes.reduce((a, b) => a + b, 0) / workerTimes.length;
      }
    }

    // Add to recent executions
    this.metrics.recentExecutions.push({
      mandateId,
      status: success ? 'completed' : 'failed',
      duration,
      timestamp: Date.now(),
    });

    // Keep only recent executions
    if (this.metrics.recentExecutions.length > this.maxRecentExecutions) {
      this.metrics.recentExecutions = this.metrics.recentExecutions.slice(-this.maxRecentExecutions);
    }
  }

  /**
   * Get current metrics.
   */
  getMetrics(): ExecutionMetrics {
    return { ...this.metrics };
  }

  /**
   * Reset metrics (for testing).
   */
  reset(): void {
    this.metrics = {
      totalMandatesProcessed: 0,
      totalMandatesFailed: 0,
      totalMandatesCompleted: 0,
      averageExecutionTime: 0,
      executionTimes: [],
      successRate: 0,
      failureRate: 0,
      byStatus: {},
      byWorker: {},
      recentExecutions: [],
    };
  }
}

