/**
 * Execution Governor Configuration
 * 
 * Configuration interface and defaults for the execution governor service.
 */

export interface GovernorConfig {
  port: number;
  maxConcurrentExecutions: number;
  maxMandatesPerMinute: number;
  defaultPriority: number;
  globalBudgetLimit: number;
  resourceLimits: {
    cpu: number;
    memory: number;
  };
  workerTimeout: number;
  retryAttempts: number;
  workerHealthCheckInterval: number;
  queueCheckInterval: number;
}

export const DEFAULT_CONFIG: GovernorConfig = {
  port: parseInt(process.env.PORT || '3000', 10),
  maxConcurrentExecutions: parseInt(process.env.MAX_CONCURRENT_EXECUTIONS || '5', 10),
  maxMandatesPerMinute: parseInt(process.env.MAX_MANDATES_PER_MINUTE || '20', 10),
  defaultPriority: 50,
  globalBudgetLimit: parseFloat(process.env.GLOBAL_BUDGET_LIMIT || '1000.0'),
  resourceLimits: {
    cpu: parseFloat(process.env.CPU_LIMIT || '2.0'),
    memory: parseFloat(process.env.MEMORY_LIMIT || '4096.0'), // MB
  },
  workerTimeout: parseInt(process.env.WORKER_TIMEOUT || '3600000', 10), // 1 hour default
  retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3', 10),
  workerHealthCheckInterval: parseInt(process.env.WORKER_HEALTH_CHECK_INTERVAL || '30000', 10), // 30 seconds
  queueCheckInterval: parseInt(process.env.QUEUE_CHECK_INTERVAL || '1000', 10), // 1 second
};

export function loadConfig(): GovernorConfig {
  return { ...DEFAULT_CONFIG };
}

