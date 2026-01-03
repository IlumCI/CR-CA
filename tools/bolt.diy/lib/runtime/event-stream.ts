/**
 * Event stream for real-time execution observability.
 * 
 * Provides Server-Sent Events (SSE) streaming for mandate execution
 * events to enable live observability dashboards.
 * 
 * Supports both direct execution events and governor-originated events.
 */

import type { ExecutionEvent } from '~/types/mandate';
import { ExecutionEventEmitter, eventRegistry } from './execution-events';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('event-stream');

/**
 * Get governor URL from environment.
 */
function getGovernorUrl(): string | null {
  if (typeof process !== 'undefined' && process.env) {
    return process.env.EXECUTION_GOVERNOR_URL || null;
  }
  return null;
}

/**
 * Create Server-Sent Events stream for mandate execution.
 * 
 * @param mandateId - The mandate ID to stream events for
 * @returns ReadableStream for SSE
 */
export function createEventStream(mandateId: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const emitter = eventRegistry.getEmitter(mandateId);

  let closed = false;

  // Send initial connection event
  const sendEvent = (event: ExecutionEvent) => {
    if (closed) return;

    const data = JSON.stringify(event);
    const sseMessage = `data: ${data}\n\n`;
    return encoder.encode(sseMessage);
  };

  // Create readable stream
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // Send initial connection message
      const initMessage = encoder.encode(`data: ${JSON.stringify({ type: 'connected', mandate_id: mandateId })}\n\n`);
      controller.enqueue(initMessage);

      // Subscribe to all events
      const unsubscribe = emitter.on('*', (event: ExecutionEvent) => {
        try {
          const encoded = sendEvent(event);
          if (encoded) {
            controller.enqueue(encoded);
          }
        } catch (error) {
          logger.error('Error sending event to stream:', error);
        }
      });

      // Handle stream closure
      const cleanup = () => {
        closed = true;
        unsubscribe();
        // Don't close controller here - let the client close it
      };

      // Store cleanup function for later
      (controller as any)._cleanup = cleanup;
    },

    cancel() {
      closed = true;
      logger.debug(`Event stream cancelled for mandate ${mandateId}`);
    },
  });

  return stream;
}

/**
 * Create WebSocket-like event stream using SSE.
 * Returns a Response object suitable for Remix/Cloudflare Pages.
 */
export function createEventStreamResponse(mandateId: string): Response {
  const stream = createEventStream(mandateId);

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no', // Disable buffering in nginx
    },
  });
}

/**
 * Send a single event to all active streams for a mandate.
 */
export function broadcastEvent(mandateId: string, event: ExecutionEvent): void {
  const emitter = eventRegistry.getEmitter(mandateId);
  emitter.emit(event.type, event.data, event.metadata);
}

/**
 * Get all events for a mandate (for polling fallback).
 */
export function getMandateEvents(mandateId: string): ExecutionEvent[] {
  const emitter = eventRegistry.getEmitter(mandateId);
  return emitter.getEvents();
}

/**
 * Get events since a specific timestamp (for polling with incremental updates).
 */
export function getMandateEventsSince(mandateId: string, since: number): ExecutionEvent[] {
  const emitter = eventRegistry.getEmitter(mandateId);
  return emitter.getEvents().filter((event) => event.timestamp > since);
}

