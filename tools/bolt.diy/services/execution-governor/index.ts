/**
 * Execution Governor Service
 * 
 * Manages mandate execution with concurrency control, priority queuing,
 * rate limiting, and resource management.
 */

import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import { loadConfig, type GovernorConfig } from './config.js';
import { MandatePriorityQueue } from './priority-queue.js';
import { MetricsCollector } from './metrics.js';
import type {
  Mandate,
  QueuedMandate,
  ActiveExecution,
  Worker,
  ExecutionStatus,
  ExecutionEvent,
  GovernorState,
} from './types.js';

const config = loadConfig();
const app = express();
app.use(cors());
app.use(express.json());

// State management
const mandateQueue = new MandatePriorityQueue();
const activeExecutions = new Map<string, ActiveExecution>();
const workers = new Map<string, Worker>();
const executionStatuses = new Map<string, ExecutionStatus>();
const metrics = new MetricsCollector();
const eventListeners = new Map<string, Set<WebSocket>>(); // mandate_id -> WebSocket clients
let globalBudgetUsed = 0;
let rateLimitWindow: { start: number; count: number } = { start: Date.now(), count: 0 };

// WebSocket server
const server = app.listen(config.port, () => {
  console.log(`Execution Governor listening on port ${config.port}`);
});

const wss = new WebSocketServer({ server });

/**
 * Calculate priority for a mandate.
 */
function calculatePriority(mandate: Mandate): number {
  let priority = config.defaultPriority;

  // Higher priority for governance-required mandates
  if (mandate.governance?.proposal_id) {
    priority += 10;
  }

  // Higher priority for critical proposals
  if (mandate.metadata?.priority === 'critical') {
    priority += 30;
  } else if (mandate.metadata?.priority === 'high') {
    priority += 20;
  } else if (mandate.metadata?.priority === 'medium') {
    priority += 10;
  }

  // Lower priority for larger budgets (more expensive)
  priority -= Math.min(20, mandate.budget.cost / 10);

  // Higher priority for earlier mandates (FIFO within same priority)
  // This is handled by queue ordering, not priority value

  return Math.max(0, Math.min(100, priority));
}

/**
 * Check if we can accept a new mandate (rate limiting).
 */
function canAcceptMandate(): boolean {
  const now = Date.now();
  const windowMs = 60 * 1000; // 1 minute

  // Reset window if expired
  if (now - rateLimitWindow.start > windowMs) {
    rateLimitWindow = { start: now, count: 0 };
  }

  // Check rate limit
  if (rateLimitWindow.count >= config.maxMandatesPerMinute) {
    return false;
  }

  // Check global budget
  if (globalBudgetUsed >= config.globalBudgetLimit) {
    return false;
  }

  return true;
}

/**
 * Check if we have available execution slots.
 */
function hasAvailableSlot(): boolean {
  return activeExecutions.size < config.maxConcurrentExecutions;
}

/**
 * Find available worker.
 */
function findAvailableWorker(): Worker | undefined {
  for (const worker of workers.values()) {
    if (
      worker.status === 'idle' &&
      worker.capabilities.currentExecutions < worker.capabilities.maxConcurrent
    ) {
      return worker;
    }
  }
  return undefined;
}

/**
 * Process queue and dispatch mandates to available workers.
 */
function processQueue(): void {
  while (!mandateQueue.isEmpty() && hasAvailableSlot()) {
    const queuedMandate = mandateQueue.peek();
    if (!queuedMandate) {
      break;
    }

    const worker = findAvailableWorker();
    if (!worker) {
      break;
    }

    // Dequeue mandate
    const mandate = mandateQueue.dequeue();
    if (!mandate) {
      break;
    }

    // Mark worker as busy
    worker.status = 'busy';
    worker.currentMandateId = mandate.mandate.mandate_id;
    worker.capabilities.currentExecutions++;

    // Create active execution
    const execution: ActiveExecution = {
      mandateId: mandate.mandate.mandate_id,
      workerId: worker.workerId,
      startedAt: Date.now(),
      estimatedCompletion: Date.now() + mandate.mandate.budget.time * 1000,
      resourceUsage: {
        cpu: config.resourceLimits.cpu / config.maxConcurrentExecutions,
        memory: config.resourceLimits.memory / config.maxConcurrentExecutions,
      },
    };
    activeExecutions.set(mandate.mandate.mandate_id, execution);

    // Update status
    const status: ExecutionStatus = {
      mandateId: mandate.mandate.mandate_id,
      status: 'executing',
      workerId: worker.workerId,
      progress: 0,
      startedAt: execution.startedAt,
    };
    executionStatuses.set(mandate.mandate.mandate_id, status);

    // Record metrics
    metrics.recordStart(mandate.mandate.mandate_id, worker.workerId);

    // Notify worker (via WebSocket or HTTP)
    broadcastToWorker(worker.workerId, {
      type: 'mandate_assigned',
      mandate: mandate.mandate,
      executionId: mandate.mandate.mandate_id,
    });

    console.log(
      `Dispatched mandate ${mandate.mandate.mandate_id} to worker ${worker.workerId}`
    );
  }
}

/**
 * Broadcast message to a specific worker.
 */
function broadcastToWorker(workerId: string, message: any): void {
  // In a real implementation, this would use WebSocket or HTTP
  // For now, workers will poll for mandates
  console.log(`Broadcasting to worker ${workerId}:`, message);
}

/**
 * Broadcast event to all listeners for a mandate.
 */
function broadcastEvent(mandateId: string, event: ExecutionEvent): void {
  const listeners = eventListeners.get(mandateId);
  if (listeners) {
    const message = JSON.stringify(event);
    for (const ws of listeners) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    }
  }
}

// ============================================================================
// HTTP API Endpoints
// ============================================================================

/**
 * POST /mandates - Accept mandate from bolt.diy API
 */
app.post('/mandates', (req: express.Request, res: express.Response) => {
  try {
    const mandate: Mandate = req.body;

    // Validate mandate
    if (!mandate.mandate_id || !mandate.objectives || !mandate.budget) {
      return res.status(400).json({ error: 'Invalid mandate structure' });
    }

    // Check rate limiting
    if (!canAcceptMandate()) {
      return res.status(429).json({
        error: 'Rate limit exceeded',
        retryAfter: 60,
      });
    }

    // Calculate priority
    const priority = calculatePriority(mandate);

    // Create queued mandate
    const queuedMandate: QueuedMandate = {
      mandate,
      priority,
      queuedAt: Date.now(),
      retryCount: 0,
    };

    // Enqueue
    mandateQueue.enqueue(queuedMandate);
    rateLimitWindow.count++;

    // Initialize status
    const status: ExecutionStatus = {
      mandateId: mandate.mandate_id,
      status: 'queued',
      queuePosition: mandateQueue.getPosition(mandate.mandate_id),
    };
    executionStatuses.set(mandate.mandate_id, status);

    // Try to process queue
    processQueue();

    res.status(202).json({
      success: true,
      mandate_id: mandate.mandate_id,
      status: 'queued',
      queue_position: status.queuePosition,
      estimated_wait_time: estimateWaitTime(),
    });
  } catch (error) {
    console.error('Error accepting mandate:', error);
    res.status(500).json({ error: 'Failed to accept mandate' });
  }
});

/**
 * GET /mandates/:id/status - Get mandate execution status
 */
app.get('/mandates/:id/status', (req, res) => {
  const status = executionStatuses.get(req.params.id);
  if (!status) {
    return res.status(404).json({ error: 'Mandate not found' });
  }

  // Update queue position if queued
  if (status.status === 'queued') {
    status.queuePosition = mandateQueue.getPosition(req.params.id);
  }

  res.json(status);
});

/**
 * POST /workers/register - Register headless worker
 */
app.post('/workers/register', (req, res) => {
  try {
    const { workerId, capabilities } = req.body;

    if (!workerId) {
      return res.status(400).json({ error: 'workerId is required' });
    }

    const worker: Worker = {
      workerId,
      registeredAt: Date.now(),
      lastHeartbeat: Date.now(),
      status: 'idle',
      capabilities: {
        maxConcurrent: capabilities?.maxConcurrent || 1,
        currentExecutions: 0,
      },
    };

    workers.set(workerId, worker);

    console.log(`Worker ${workerId} registered`);

    res.json({
      success: true,
      workerId,
      message: 'Worker registered successfully',
    });
  } catch (error) {
    console.error('Error registering worker:', error);
    res.status(500).json({ error: 'Failed to register worker' });
  }
});

/**
 * POST /workers/:id/heartbeat - Worker heartbeat
 */
app.post('/workers/:id/heartbeat', (req, res) => {
  const worker = workers.get(req.params.id);
  if (!worker) {
    return res.status(404).json({ error: 'Worker not found' });
  }

  worker.lastHeartbeat = Date.now();
  worker.status = req.body.status || worker.status;

  res.json({ success: true });
});

/**
 * POST /workers/:id/request-mandate - Worker requests next mandate
 */
app.post('/workers/:id/request-mandate', (req, res) => {
  const worker = workers.get(req.params.id);
  if (!worker) {
    return res.status(404).json({ error: 'Worker not found' });
  }

  // Check if worker is available
  if (
    worker.status !== 'idle' ||
    worker.capabilities.currentExecutions >= worker.capabilities.maxConcurrent
  ) {
    return res.json({ mandate: null, message: 'Worker not available' });
  }

  // Process queue to dispatch mandates
  processQueue();

  // Check if this worker was assigned a mandate
  if (worker.currentMandateId) {
    const execution = activeExecutions.get(worker.currentMandateId);
    if (execution) {
      const status = executionStatuses.get(execution.mandateId);
      // Get mandate from queue or active execution
      // For now, return the mandate ID - worker should fetch full mandate
      return res.json({
        mandate_id: execution.mandateId,
        execution_id: execution.mandateId,
      });
    }
  }

  res.json({ mandate: null, message: 'No mandate available' });
});

/**
 * GET /mandates/:id - Get full mandate details
 */
app.get('/mandates/:id', (req, res) => {
  const status = executionStatuses.get(req.params.id);
  if (!status) {
    return res.status(404).json({ error: 'Mandate not found' });
  }

  // Find mandate in queue or active execution
  const queued = mandateQueue.get(req.params.id);
  if (queued) {
    return res.json({ mandate: queued.mandate });
  }

  const execution = activeExecutions.get(req.params.id);
  if (execution) {
    // Mandate is being executed - we'd need to store it
    // For now, return error
    return res.status(404).json({ error: 'Mandate not available (in execution)' });
  }

  res.status(404).json({ error: 'Mandate not found' });
});

/**
 * POST /workers/:id/report-progress - Worker reports execution progress
 */
app.post('/workers/:id/report-progress', (req, res) => {
  const worker = workers.get(req.params.id);
  if (!worker) {
    return res.status(404).json({ error: 'Worker not found' });
  }

  const { mandateId, event, progress, status: execStatus } = req.body;

  if (!mandateId) {
    return res.status(400).json({ error: 'mandateId is required' });
  }

  const execution = activeExecutions.get(mandateId);
  if (!execution || execution.workerId !== req.params.id) {
    return res.status(404).json({ error: 'Execution not found' });
  }

  // Update execution status
  const status = executionStatuses.get(mandateId);
  if (status) {
    if (progress !== undefined) {
      status.progress = progress;
    }
    if (execStatus) {
      status.status = execStatus;
    }
  }

  // Broadcast event if provided
  if (event) {
    const executionEvent: ExecutionEvent = {
      mandate_id: mandateId,
      iteration: event.iteration || 0,
      type: event.type,
      timestamp: event.timestamp || Date.now(),
      data: event.data || {},
      metadata: event.metadata || {},
    };
    broadcastEvent(mandateId, executionEvent);
  }

  res.json({ success: true });
});

/**
 * POST /workers/:id/complete - Worker reports mandate completion
 */
app.post('/workers/:id/complete', (req, res) => {
  const worker = workers.get(req.params.id);
  if (!worker) {
    return res.status(404).json({ error: 'Worker not found' });
  }

  const { mandateId, success, result, error } = req.body;

  if (!mandateId) {
    return res.status(400).json({ error: 'mandateId is required' });
  }

  const execution = activeExecutions.get(mandateId);
  if (!execution || execution.workerId !== req.params.id) {
    return res.status(404).json({ error: 'Execution not found' });
  }

  // Calculate duration
  const duration = Date.now() - execution.startedAt;

  // Update status
  const status = executionStatuses.get(mandateId);
  if (status) {
    status.status = success ? 'completed' : 'failed';
    status.completedAt = Date.now();
    status.result = result;
    status.error = error;
  }

  // Record metrics
  metrics.recordCompletion(mandateId, worker.workerId, duration, success);

  // Update budget
  if (result?.budget_summary?.cost_incurred) {
    globalBudgetUsed += result.budget_summary.cost_incurred;
  }

  // Clean up
  activeExecutions.delete(mandateId);
  worker.status = 'idle';
  worker.currentMandateId = undefined;
  worker.capabilities.currentExecutions--;

  // Broadcast completion event
  const completionEvent: ExecutionEvent = {
    mandate_id: mandateId,
    iteration: 0,
    type: success ? 'iteration_end' : 'error',
    timestamp: Date.now(),
    data: {
      status: success ? 'success' : 'failed',
      ...(error && { message: error }),
    },
    metadata: {},
  };
  broadcastEvent(mandateId, completionEvent);

  // Process queue for next mandate
  processQueue();

  res.json({ success: true });

  console.log(
    `Mandate ${mandateId} ${success ? 'completed' : 'failed'} by worker ${worker.workerId} in ${duration}ms`
  );
});

/**
 * GET /governor/state - Get governor state
 */
app.get('/governor/state', (req, res) => {
  const healthyWorkers = Array.from(workers.values()).filter(
    (w) => Date.now() - w.lastHeartbeat < config.workerHealthCheckInterval * 2
  ).length;

  const state: GovernorState = {
    queueDepth: mandateQueue.size(),
    activeExecutions: activeExecutions.size,
    maxConcurrentExecutions: config.maxConcurrentExecutions,
    registeredWorkers: workers.size,
    healthyWorkers,
    totalMandatesProcessed: metrics.getMetrics().totalMandatesProcessed,
    totalMandatesFailed: metrics.getMetrics().totalMandatesFailed,
    averageExecutionTime: metrics.getMetrics().averageExecutionTime,
    currentBudgetUsage: globalBudgetUsed,
    globalBudgetLimit: config.globalBudgetLimit,
  };

  res.json(state);
});

/**
 * GET /governor/metrics - Get execution metrics
 */
app.get('/governor/metrics', (req, res) => {
  res.json(metrics.getMetrics());
});

/**
 * Estimate wait time for a mandate in queue.
 */
function estimateWaitTime(): number {
  // Simple estimation: average execution time * queue position
  const avgTime = metrics.getMetrics().averageExecutionTime || 60000; // 1 minute default
  const queueDepth = mandateQueue.size();
  const activeCount = activeExecutions.size;
  const availableSlots = config.maxConcurrentExecutions - activeCount;

  if (availableSlots > 0) {
    return 0; // Can start immediately
  }

  // Estimate based on current executions completing
  return Math.ceil((queueDepth / config.maxConcurrentExecutions) * avgTime);
}

// ============================================================================
// WebSocket Server
// ============================================================================

wss.on('connection', (ws: WebSocket, req) => {
  console.log('WebSocket client connected');

  ws.on('message', (message: string) => {
    try {
      const data = JSON.parse(message.toString());
      
      if (data.type === 'subscribe') {
        // Subscribe to events for a mandate
        const mandateId = data.mandate_id;
        if (mandateId) {
          if (!eventListeners.has(mandateId)) {
            eventListeners.set(mandateId, new Set());
          }
          eventListeners.get(mandateId)!.add(ws);
          ws.send(JSON.stringify({ type: 'subscribed', mandate_id: mandateId }));
        }
      } else if (data.type === 'unsubscribe') {
        // Unsubscribe from mandate events
        const mandateId = data.mandate_id;
        if (mandateId && eventListeners.has(mandateId)) {
          eventListeners.get(mandateId)!.delete(ws);
        }
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  });

  ws.on('close', () => {
    // Remove from all event listeners
    for (const listeners of eventListeners.values()) {
      listeners.delete(ws);
    }
    console.log('WebSocket client disconnected');
  });
});

// ============================================================================
// Worker Health Monitoring
// ============================================================================

setInterval(() => {
  const now = Date.now();
  for (const [workerId, worker] of workers.entries()) {
    const timeSinceHeartbeat = now - worker.lastHeartbeat;
    if (timeSinceHeartbeat > config.workerHealthCheckInterval * 2) {
      console.warn(`Worker ${workerId} appears unhealthy (no heartbeat for ${timeSinceHeartbeat}ms)`);
      worker.status = 'unhealthy';

      // If worker has active execution, mark it for retry
      if (worker.currentMandateId) {
        const execution = activeExecutions.get(worker.currentMandateId);
        if (execution) {
          // Re-queue mandate for retry
          const status = executionStatuses.get(execution.mandateId);
          if (status && status.status === 'executing') {
            // Find original mandate (would need to store it)
            // For now, just mark execution as failed
            status.status = 'failed';
            status.error = 'Worker became unhealthy';
            activeExecutions.delete(execution.mandateId);
            worker.status = 'idle';
            worker.currentMandateId = undefined;
            worker.capabilities.currentExecutions--;
            processQueue();
          }
        }
      }
    } else if (worker.status === 'unhealthy' && timeSinceHeartbeat < config.workerHealthCheckInterval) {
      // Worker recovered
      worker.status = 'idle';
      console.log(`Worker ${workerId} recovered`);
    }
  }
}, config.workerHealthCheckInterval);

// ============================================================================
// Queue Processing
// ============================================================================

setInterval(() => {
  processQueue();
}, config.queueCheckInterval);

// ============================================================================
// Graceful Shutdown
// ============================================================================

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  wss.close(() => {
    server.close(() => {
      console.log('Execution Governor shut down');
      process.exit(0);
    });
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  wss.close(() => {
    server.close(() => {
      console.log('Execution Governor shut down');
      process.exit(0);
    });
  });
});

