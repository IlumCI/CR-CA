/**
 * Mandate API endpoint for autonomous code generation.
 * 
 * Receives structured mandates from CorporateSwarm and executes
 * code generation cycles autonomously with governance oversight.
 */

import { type ActionFunctionArgs } from '@remix-run/cloudflare';
import { createScopedLogger } from '~/utils/logger';
import type { Mandate, ExecutionResult } from '~/types/mandate';
import { eventRegistry } from '~/lib/runtime/execution-events';
import { createEventStreamResponse, broadcastEvent } from '~/lib/runtime/event-stream';

const logger = createScopedLogger('api.mandate');

/**
 * Validate mandate structure and constraints.
 */
function validateMandate(mandate: Mandate): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Validate mandate_id
  if (!mandate.mandate_id || typeof mandate.mandate_id !== 'string') {
    errors.push('mandate_id is required and must be a string');
  }

  // Validate objectives
  if (!Array.isArray(mandate.objectives) || mandate.objectives.length === 0) {
    errors.push('objectives must be a non-empty array');
  }

  // Validate constraints
  if (!mandate.constraints) {
    errors.push('constraints are required');
  } else {
    if (mandate.constraints.maxDependencies < 0) {
      errors.push('maxDependencies must be non-negative');
    }
    if (mandate.constraints.maxFileSize < 0) {
      errors.push('maxFileSize must be non-negative');
    }
    if (mandate.constraints.maxFiles < 0) {
      errors.push('maxFiles must be non-negative');
    }
  }

  // Validate budget
  if (!mandate.budget) {
    errors.push('budget is required');
  } else {
    if (mandate.budget.token < 0) {
      errors.push('budget.token must be non-negative');
    }
    if (mandate.budget.time < 0) {
      errors.push('budget.time must be non-negative');
    }
    if (mandate.budget.cost < 0) {
      errors.push('budget.cost must be non-negative');
    }
  }

  // Validate deliverables
  if (!Array.isArray(mandate.deliverables) || mandate.deliverables.length === 0) {
    errors.push('deliverables must be a non-empty array');
  }

  // Validate iteration_config
  if (!mandate.iteration_config) {
    errors.push('iteration_config is required');
  } else {
    if (mandate.iteration_config.max_iterations < 1) {
      errors.push('max_iterations must be at least 1');
    }
    if (
      mandate.iteration_config.quality_threshold < 0 ||
      mandate.iteration_config.quality_threshold > 1
    ) {
      errors.push('quality_threshold must be between 0 and 1');
    }
  }

  // Validate deployment config if enabled
  if (mandate.deployment?.enabled) {
    if (!mandate.deployment.provider) {
      errors.push('deployment.provider is required when deployment is enabled');
    } else {
      const validProviders = ['netlify', 'vercel', 'github', 'gitlab'];
      if (!validProviders.includes(mandate.deployment.provider)) {
        errors.push(`deployment.provider must be one of: ${validProviders.join(', ')}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Main mandate action handler.
 */
export async function action(args: ActionFunctionArgs) {
  return mandateAction(args);
}

async function mandateAction({ context, request }: ActionFunctionArgs) {
  try {
    // Parse mandate from request
    const mandate: Mandate = await request.json();

    logger.info(`Received mandate: ${mandate.mandate_id}`);

    // Validate mandate
    const validation = validateMandate(mandate);
    if (!validation.valid) {
      logger.error(`Mandate validation failed: ${validation.errors.join(', ')}`);
      return new Response(
        JSON.stringify({
          success: false,
          error: 'Invalid mandate',
          errors: validation.errors,
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Check if this is a request for event stream
    const url = new URL(request.url);
    const streamRequested = url.searchParams.get('stream') === 'true';

    if (streamRequested) {
      // Return event stream for real-time observability
      return createEventStreamResponse(mandate.mandate_id);
    }

    // Check if Execution Governor is enabled
    const governorEnabled = process.env.EXECUTION_GOVERNOR_ENABLED === 'true' || 
                           process.env.EXECUTION_GOVERNOR_URL !== undefined;
    const governorUrl = process.env.EXECUTION_GOVERNOR_URL || 'http://localhost:3000';

    if (governorEnabled) {
      // Forward mandate to Execution Governor
      try {
        logger.info(`Forwarding mandate ${mandate.mandate_id} to Execution Governor at ${governorUrl}`);
        
        const governorResponse = await fetch(`${governorUrl}/mandates`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(mandate),
        });

        if (!governorResponse.ok) {
          const errorText = await governorResponse.text();
          logger.error(`Governor rejected mandate: ${governorResponse.status} - ${errorText}`);
          
          // Fall back to direct execution if governor fails
          logger.warn('Falling back to direct execution mode');
        } else {
          const governorData = await governorResponse.json() as { queue_position?: number; estimated_wait_time?: number };
          logger.info(`Mandate ${mandate.mandate_id} queued in governor:`, governorData);

          // Initialize event emitter for observability
          const emitter = eventRegistry.getEmitter(mandate.mandate_id);
          
          // Store mandate for later retrieval
          eventRegistry.storeMandate(mandate.mandate_id, mandate);
          
          emitter.emitLog('info', `Mandate ${mandate.mandate_id} queued in Execution Governor`, 'api');

          return new Response(
            JSON.stringify({
              success: true,
              mandate_id: mandate.mandate_id,
              status: 'queued',
              message: 'Mandate queued in Execution Governor',
              queue_position: governorData.queue_position,
              estimated_wait_time: governorData.estimated_wait_time,
              event_stream_url: `/api/mandate?stream=true&mandate_id=${mandate.mandate_id}`,
              governor_url: governorUrl,
            }),
            {
              status: 202, // Accepted
              headers: { 'Content-Type': 'application/json' },
            }
          );
        }
      } catch (error) {
        logger.error('Error forwarding to governor, falling back to direct execution:', error);
        // Fall through to direct execution
      }
    }

    // Direct execution mode (fallback or when governor disabled)
    // Initialize event emitter for this mandate
    const emitter = eventRegistry.getEmitter(mandate.mandate_id);
    
    // Store mandate for later retrieval
    eventRegistry.storeMandate(mandate.mandate_id, mandate);

    // Emit initial acceptance event
    emitter.emitIterationStart(0, {
      model_used: 'pending',
    });

    emitter.emitLog('info', `Mandate ${mandate.mandate_id} accepted and queued for execution`, 'api');

    // Note: Actual execution will be handled client-side where WebContainer is available
    // The mandate is validated and accepted here, execution happens via client-side MandateExecutor
    logger.info(`Mandate ${mandate.mandate_id} accepted, ready for client-side execution`);

    return new Response(
      JSON.stringify({
        success: true,
        mandate_id: mandate.mandate_id,
        status: 'accepted',
        message: 'Mandate accepted, execution started',
        event_stream_url: `/api/mandate?stream=true&mandate_id=${mandate.mandate_id}`,
      }),
      {
        status: 202, // Accepted
        headers: { 'Content-Type': 'application/json' },
      }
    );
  } catch (error: unknown) {
    logger.error('Error processing mandate:', error);

    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    return new Response(
      JSON.stringify({
        success: false,
        error: 'Failed to process mandate',
        message: errorMessage,
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

/**
 * GET handler for mandate status and event polling.
 */
export async function loader({ request }: ActionFunctionArgs) {
  const url = new URL(request.url);
  
  // Check if this is a request for active mandates list (no mandate_id required)
  if (url.searchParams.get('list') === 'true') {
    const allMandateIds = eventRegistry.getActiveMandates();
    const mandates = allMandateIds.slice(-10).reverse().map(mandateId => {
      const emitter = eventRegistry.getEmitter(mandateId);
      const events = emitter.getEvents();
      const latestEvent = events[events.length - 1];
      
      return {
        mandate_id: mandateId,
        last_event_time: latestEvent?.timestamp || Date.now() / 1000,
        event_count: events.length,
        status: latestEvent?.type === 'iteration_end' ? 'completed' : 
                latestEvent?.type === 'error' ? 'failed' :
                events.length > 0 ? 'running' : 'accepted'
      };
    });

    return new Response(
      JSON.stringify({
        mandates,
        count: mandates.length,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  // For other requests, mandate_id is required
  const mandateId = url.searchParams.get('mandate_id');

  if (!mandateId) {
    return new Response(
      JSON.stringify({
        error: 'mandate_id parameter is required (or use ?list=true for mandates list)',
      }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  // Check if this is a request for the mandate object itself
  if (url.searchParams.get('get') === 'true') {
    const storedMandate = eventRegistry.getMandate(mandateId);
    if (storedMandate) {
      return new Response(
        JSON.stringify({
          mandate: storedMandate,
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    } else {
      return new Response(
        JSON.stringify({
          error: 'Mandate not found',
        }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }

  // Check if streaming is requested
  const streamRequested = url.searchParams.get('stream') === 'true';
  if (streamRequested) {
    return createEventStreamResponse(mandateId);
  }

  // Return current events for polling
  const { getMandateEvents, getMandateEventsSince } = await import('~/lib/runtime/event-stream');
  const since = url.searchParams.get('since');
  const events = since
    ? getMandateEventsSince(mandateId, parseInt(since, 10))
    : getMandateEvents(mandateId);

  return new Response(
    JSON.stringify({
      mandate_id: mandateId,
      events,
      count: events.length,
    }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

