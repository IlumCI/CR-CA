/**
 * Auto-execution page for headless workers.
 * 
 * This page automatically executes a mandate when loaded, designed for
 * use by headless browser workers controlled by Playwright.
 */

import { json, type LoaderFunctionArgs, type MetaFunction } from '@remix-run/cloudflare';
import { useLoaderData, useParams } from '@remix-run/react';
import { useEffect, useRef } from 'react';
import type { Mandate, ExecutionResult } from '~/types/mandate';
import { MandateExecutor } from '~/lib/runtime/mandate-executor';
import { webcontainer } from '~/lib/webcontainer';
import { newBoltShellProcess } from '~/utils/shell';
import { createScopedLogger } from '~/utils/logger';
import type { IProviderSetting } from '~/types/model';

const logger = createScopedLogger('execute-page');

export const meta: MetaFunction<typeof loader> = () => {
  return [
    { title: 'Mandate Execution' },
    { name: 'robots', content: 'noindex, nofollow' },
  ];
};

const GOVERNOR_URL = typeof window !== 'undefined' 
  ? (window as any).__GOVERNOR_URL__ || process.env.EXECUTION_GOVERNOR_URL || 'http://localhost:3000'
  : 'http://localhost:3000';

/**
 * Load mandate from governor or window injection.
 */
export async function loader({ params, request }: LoaderFunctionArgs) {
  const mandateId = params.mandateId;
  
  if (!mandateId) {
    throw new Response('Mandate ID is required', { status: 400 });
  }

  // Try to load mandate from governor
  try {
    const governorUrl = process.env.EXECUTION_GOVERNOR_URL || 'http://localhost:3000';
    const response = await fetch(`${governorUrl}/mandates/${mandateId}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.ok) {
      const data = await response.json() as { mandate?: Mandate };
      return json({ mandateId, mandate: data.mandate || null });
    }
  } catch (error) {
    logger.warn('Could not load mandate from governor, will use window injection:', error);
  }

  return json({ mandateId, mandate: null });
}

/**
 * Auto-execution component.
 */
function AutoExecutor({ mandateId, mandate: initialMandate }: { mandateId: string; mandate: Mandate | null }) {
  const executedRef = useRef(false);
  const resultRef = useRef<ExecutionResult | null>(null);
  const errorRef = useRef<Error | null>(null);

  useEffect(() => {
    if (executedRef.current) {
      return; // Already executed
    }

    executedRef.current = true;

    const execute = async () => {
      try {
        // Get mandate from window injection (Playwright) or use loaded mandate
        let mandate: Mandate | null = initialMandate || null;

        if (!mandate && typeof window !== 'undefined') {
          mandate = (window as any).__MANDATE_DATA__ || null;
        }

        if (!mandate) {
          // Try to fetch from governor
          try {
            const response = await fetch(`${GOVERNOR_URL}/mandates/${mandateId}`);
            if (response.ok) {
              const data = await response.json() as { mandate?: Mandate };
              mandate = data.mandate || null;
            }
          } catch (error) {
            logger.error('Failed to fetch mandate:', error);
          }
        }

        if (!mandate) {
          throw new Error(`Mandate ${mandateId} not found`);
        }

        logger.info(`Starting auto-execution of mandate ${mandateId}`);

        // Set execution status in DOM for Playwright to detect
        const statusEl = document.createElement('div');
        statusEl.id = 'execution-status';
        statusEl.setAttribute('data-execution-status', 'initializing');
        statusEl.style.display = 'none';
        document.body.appendChild(statusEl);

        // Fetch API keys and provider settings
        const apiKeysResponse = await fetch('/api/export-api-keys');
        const apiKeys: Record<string, string> = apiKeysResponse.ok 
          ? await apiKeysResponse.json() 
          : {};

        // Fetch provider settings
        const providersResponse = await fetch('/api/configured-providers');
        const providersData = providersResponse.ok 
          ? await providersResponse.json() as { providers?: Array<{ name: string; [key: string]: any }> }
          : { providers: [] };
        
        const providerSettings: Record<string, IProviderSetting> = {};
        if (providersData.providers) {
          for (const provider of providersData.providers) {
            providerSettings[provider.name] = provider as IProviderSetting;
          }
        }

        // Initialize WebContainer
        const wc = await webcontainer;
        logger.info('WebContainer initialized');

        // Create shell terminal
        const shellTerminal = () => newBoltShellProcess();

        // Create MandateExecutor
        const executor = new MandateExecutor(
          mandate,
          Promise.resolve(wc),
          shellTerminal,
          apiKeys,
          providerSettings
        );

        // Set up event forwarding to governor
        const eventEmitter = (executor as any).eventEmitter;
        if (eventEmitter) {
          eventEmitter.on('*', (event: any) => {
            // Forward event to governor
            fetch(`${GOVERNOR_URL}/workers/${process.env.WORKER_ID || 'headless'}/report-progress`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                mandateId: mandate.mandate_id,
                event: {
                  mandate_id: event.mandate_id,
                  iteration: event.iteration,
                  type: event.type,
                  timestamp: event.timestamp,
                  data: event.data,
                  metadata: event.metadata,
                },
              }),
            }).catch((error) => {
              logger.error('Failed to forward event to governor:', error);
            });
          });
        }

        // Update status
        statusEl.setAttribute('data-execution-status', 'executing');

        // Execute mandate
        const result = await executor.execute();

        // Store result
        resultRef.current = result;
        if (typeof window !== 'undefined') {
          (window as any).__EXECUTION_RESULT__ = result;
        }

        // Update status
        statusEl.setAttribute('data-execution-status', 'completed');

        // Log completion
        logger.info(`Mandate ${mandateId} execution completed: ${result.status}`);
        console.log('EXECUTION_COMPLETE', result);

        // Report to governor
        await fetch(`${GOVERNOR_URL}/workers/${process.env.WORKER_ID || 'headless'}/complete`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mandateId: mandate.mandate_id,
            success: result.status === 'success',
            result,
          }),
        }).catch((error) => {
          logger.error('Failed to report completion to governor:', error);
        });
      } catch (error) {
        logger.error(`Error executing mandate ${mandateId}:`, error);
        errorRef.current = error instanceof Error ? error : new Error(String(error));

        // Store error
        if (typeof window !== 'undefined') {
          (window as any).__EXECUTION_ERROR__ = error instanceof Error ? error.message : String(error);
        }

        // Update status
        const statusEl = document.getElementById('execution-status');
        if (statusEl) {
          statusEl.setAttribute('data-execution-status', 'failed');
        }

        // Log error
        console.error('EXECUTION_ERROR', error);

        // Report failure to governor
        await fetch(`${GOVERNOR_URL}/workers/${process.env.WORKER_ID || 'headless'}/complete`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mandateId,
            success: false,
            error: error instanceof Error ? error.message : String(error),
          }),
        }).catch((err) => {
          logger.error('Failed to report failure to governor:', err);
        });
      }
    };

    execute();
  }, [mandateId, initialMandate]);

  return (
    <div className="flex flex-col h-screen w-full bg-bolt-elements-background-depth-1">
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-bolt-elements-textPrimary mb-4">
            Executing Mandate
          </h1>
          <p className="text-bolt-elements-textSecondary mb-2">
            Mandate ID: {mandateId}
          </p>
          <div className="mt-4">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-accent-500"></div>
          </div>
          <p className="text-sm text-bolt-elements-textTertiary mt-4">
            This page is executing the mandate automatically.
            <br />
            Execution status is being reported to the governor.
          </p>
        </div>
      </div>
      
      {/* Hidden status element for Playwright detection */}
      <div
        id="execution-status"
        data-execution-status="initializing"
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default function ExecuteRoute() {
  const { mandateId, mandate } = useLoaderData<typeof loader>();
  const params = useParams();
  const finalMandateId = mandateId || params.mandateId;

  if (!finalMandateId) {
    return (
      <div className="flex flex-col h-screen w-full bg-bolt-elements-background-depth-1 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-bolt-elements-textPrimary mb-4">Error</h1>
          <p className="text-bolt-elements-textSecondary">Mandate ID is required</p>
        </div>
      </div>
    );
  }

  return <AutoExecutor mandateId={finalMandateId} mandate={mandate} />;
}

