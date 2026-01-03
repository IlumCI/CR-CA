/**
 * Headless Executor Worker Service
 * 
 * Connects to Execution Governor and executes mandates using Playwright.
 */

import { WebSocket } from 'ws';
import { HeadlessExecutor, type ExecutorConfig } from './executor.js';
import type { Mandate, ExecutionEvent } from './types.js';

const GOVERNOR_URL = process.env.GOVERNOR_URL || 'http://localhost:3000';
const BOLT_DIY_URL = process.env.BOLT_DIY_URL || 'http://localhost:5173';
const WORKER_ID = process.env.WORKER_ID || `worker-${Date.now()}`;
const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '1', 10);
const POLL_INTERVAL = parseInt(process.env.POLL_INTERVAL || '5000', 10);

const config: ExecutorConfig = {
  boltDiyUrl: BOLT_DIY_URL,
  governorUrl: GOVERNOR_URL,
  workerId: WORKER_ID,
  headless: process.env.HEADLESS !== 'false',
  timeout: parseInt(process.env.EXECUTION_TIMEOUT || '3600000', 10), // 1 hour default
};

const executor = new HeadlessExecutor(config);
let isRunning = false;
let currentExecutions = 0;
let ws: WebSocket | null = null;

/**
 * Register worker with governor.
 */
async function registerWorker(): Promise<boolean> {
  try {
    const response = await fetch(`${GOVERNOR_URL}/workers/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        workerId: WORKER_ID,
        capabilities: {
          maxConcurrent: MAX_CONCURRENT,
        },
      }),
    });

    if (!response.ok) {
      console.error(`Failed to register worker: ${response.statusText}`);
      return false;
    }

    const result = await response.json();
    console.log(`Worker registered: ${result.message}`);
    return true;
  } catch (error) {
    console.error('Error registering worker:', error);
    return false;
  }
}

/**
 * Send heartbeat to governor.
 */
async function sendHeartbeat(): Promise<void> {
  try {
    await fetch(`${GOVERNOR_URL}/workers/${WORKER_ID}/heartbeat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: currentExecutions > 0 ? 'busy' : 'idle',
      }),
    });
  } catch (error) {
    console.error('Error sending heartbeat:', error);
  }
}

/**
 * Request next mandate from governor.
 */
async function requestMandate(): Promise<Mandate | null> {
  try {
    const response = await fetch(`${GOVERNOR_URL}/workers/${WORKER_ID}/request-mandate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    if (!data.mandate_id) {
      return null;
    }

    // Fetch full mandate details
    const mandateResponse = await fetch(`${GOVERNOR_URL}/mandates/${data.mandate_id}`);
    if (!mandateResponse.ok) {
      return null;
    }

    const mandateData = await mandateResponse.json();
    return mandateData.mandate;
  } catch (error) {
    console.error('Error requesting mandate:', error);
    return null;
  }
}

/**
 * Report execution progress to governor.
 */
async function reportProgress(
  mandateId: string,
  event: ExecutionEvent,
  progress?: number
): Promise<void> {
  try {
    await fetch(`${GOVERNOR_URL}/workers/${WORKER_ID}/report-progress`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mandateId,
        event,
        progress,
      }),
    });
  } catch (error) {
    console.error('Error reporting progress:', error);
  }
}

/**
 * Report execution completion to governor.
 */
async function reportCompletion(
  mandateId: string,
  success: boolean,
  result?: any,
  error?: string
): Promise<void> {
  try {
    await fetch(`${GOVERNOR_URL}/workers/${WORKER_ID}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mandateId,
        success,
        result,
        error,
      }),
    });
  } catch (error) {
    console.error('Error reporting completion:', error);
  }
}

/**
 * Execute a mandate.
 */
async function executeMandate(mandate: Mandate): Promise<void> {
  currentExecutions++;
  const startTime = Date.now();

  console.log(`Starting execution of mandate ${mandate.mandate_id}`);

  try {
    // Initialize executor if needed
    await executor.initialize();

    // Report start
    await reportProgress(mandate.mandate_id, {
      mandate_id: mandate.mandate_id,
      iteration: 0,
      type: 'iteration_start',
      timestamp: Date.now(),
      data: { status: 'running' },
      metadata: {},
    });

    // Execute mandate
    const result = await executor.executeMandate(mandate);

    const duration = Date.now() - startTime;
    console.log(
      `Mandate ${mandate.mandate_id} ${result.status} in ${duration}ms`
    );

    // Report completion
    await reportCompletion(mandate.mandate_id, result.status === 'success', result);
  } catch (error) {
    const duration = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Mandate ${mandate.mandate_id} failed:`, errorMessage);

    // Report failure
    await reportCompletion(mandate.mandate_id, false, undefined, errorMessage);
  } finally {
    currentExecutions--;
  }
}

/**
 * Main worker loop.
 */
async function workerLoop(): Promise<void> {
  while (isRunning) {
    try {
      // Check if we can accept more work
      if (currentExecutions >= MAX_CONCURRENT) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
        continue;
      }

      // Request mandate
      const mandate = await requestMandate();
      if (!mandate) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
        continue;
      }

      // Execute mandate (don't await - run in parallel)
      executeMandate(mandate).catch((error) => {
        console.error(`Error in mandate execution:`, error);
      });
    } catch (error) {
      console.error('Error in worker loop:', error);
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
    }
  }
}

/**
 * Connect to governor WebSocket for real-time updates.
 */
function connectWebSocket(): void {
  const wsUrl = GOVERNOR_URL.replace('http://', 'ws://').replace('https://', 'wss://');
  ws = new WebSocket(wsUrl);

  ws.on('open', () => {
    console.log('Connected to governor WebSocket');
  });

  ws.on('message', (data: Buffer) => {
    try {
      const message = JSON.parse(data.toString());
      if (message.type === 'mandate_assigned' && message.mandate) {
        // Governor pushed a mandate - execute it
        if (currentExecutions < MAX_CONCURRENT) {
          executeMandate(message.mandate).catch((error) => {
            console.error(`Error executing pushed mandate:`, error);
          });
        }
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  ws.on('close', () => {
    console.log('WebSocket closed, reconnecting...');
    setTimeout(connectWebSocket, 5000);
  });
}

/**
 * Initialize and start worker.
 */
async function start(): Promise<void> {
  console.log(`Starting headless executor worker ${WORKER_ID}`);
  console.log(`Governor URL: ${GOVERNOR_URL}`);
  console.log(`Bolt.diy URL: ${BOLT_DIY_URL}`);
  console.log(`Max concurrent: ${MAX_CONCURRENT}`);

  // Register with governor
  const registered = await registerWorker();
  if (!registered) {
    console.error('Failed to register worker, exiting');
    process.exit(1);
  }

  // Connect WebSocket
  connectWebSocket();

  // Start heartbeat
  setInterval(sendHeartbeat, 30000); // Every 30 seconds

  // Start worker loop
  isRunning = true;
  workerLoop().catch((error) => {
    console.error('Worker loop error:', error);
    process.exit(1);
  });
}

/**
 * Graceful shutdown.
 */
async function shutdown(): Promise<void> {
  console.log('Shutting down worker...');
  isRunning = false;

  if (ws) {
    ws.close();
  }

  await executor.cleanup();
  process.exit(0);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Start worker
start().catch((error) => {
  console.error('Failed to start worker:', error);
  process.exit(1);
});

