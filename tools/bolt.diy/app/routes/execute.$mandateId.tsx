/**
 * Auto-execution page for headless workers.
 * 
 * This page automatically executes a mandate when loaded, designed for
 * use by headless browser workers controlled by Playwright.
 */

import { json, type LoaderFunctionArgs, type MetaFunction } from '@remix-run/cloudflare';
import { useLoaderData, useParams } from '@remix-run/react';
import { useEffect, useRef, useState } from 'react';
import type { Mandate, ExecutionResult } from '~/types/mandate';
import { MandateExecutor } from '~/lib/runtime/mandate-executor';
import { webcontainer } from '~/lib/webcontainer';
import { newBoltShellProcess } from '~/utils/shell';
import { createScopedLogger } from '~/utils/logger';
import type { IProviderSetting } from '~/types/model';
import { eventRegistry } from '~/lib/runtime/execution-events';

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
  const [currentPhase, setCurrentPhase] = useState<string>('initializing');
  const [recentLogs, setRecentLogs] = useState<Array<{ level: string; message: string; timestamp: number }>>([]);

  useEffect(() => {
    if (executedRef.current) {
      return; // Already executed
    }

    executedRef.current = true;

    const execute = async () => {
      const initStartTime = Date.now();
      const eventEmitter = eventRegistry.getEmitter(mandateId);
      
      try {
        setCurrentPhase('loading_mandate');
        eventEmitter.emitInitializationStart({ phase: 'loading_mandate', timestamp: initStartTime });
        eventEmitter.emitLog('info', 'Starting mandate execution initialization', 'execute-page');
        
        // Get mandate from window injection (Playwright) or use loaded mandate
        let mandate: Mandate | null = initialMandate || null;
        const mandateLoadStart = Date.now();

        if (!mandate && typeof window !== 'undefined') {
          eventEmitter.emitLog('debug', 'Checking window injection for mandate data', 'execute-page');
          mandate = (window as any).__MANDATE_DATA__ || null;
        }

        if (!mandate) {
          // Try to fetch from API endpoint first (where mandates are stored)
          eventEmitter.emitLog('debug', `Fetching mandate from API endpoint`, 'execute-page');
          try {
            const apiResponse = await fetch(`/api/mandate?mandate_id=${mandateId}&get=true`);
            if (apiResponse.ok) {
              const apiData = await apiResponse.json() as { mandate?: Mandate };
              mandate = apiData.mandate || null;
              if (mandate) {
                eventEmitter.emitLog('info', 'Mandate loaded from API endpoint', 'execute-page');
              }
            }
          } catch (error) {
            logger.error('Failed to fetch mandate from API:', error);
            eventEmitter.emitLog('warn', `Failed to fetch from API: ${error instanceof Error ? error.message : String(error)}`, 'execute-page');
          }
        }

        if (!mandate) {
          // Try to fetch from governor as fallback
          eventEmitter.emitLog('debug', `Fetching mandate from governor at ${GOVERNOR_URL}`, 'execute-page');
          try {
            const response = await fetch(`${GOVERNOR_URL}/mandates/${mandateId}`);
            if (response.ok) {
              const data = await response.json() as { mandate?: Mandate };
              mandate = data.mandate || null;
              if (mandate) {
                eventEmitter.emitLog('info', 'Mandate loaded from governor', 'execute-page');
              }
            }
          } catch (error) {
            logger.error('Failed to fetch mandate from governor:', error);
            eventEmitter.emitLog('warn', `Failed to fetch from governor: ${error instanceof Error ? error.message : String(error)}`, 'execute-page');
          }
        }

        if (!mandate) {
          const errorMsg = `Mandate ${mandateId} not found`;
          eventEmitter.emitError(errorMsg, 'execute-page');
          throw new Error(errorMsg);
        }

        const mandateLoadTime = Date.now() - mandateLoadStart;
        eventEmitter.emitLog('info', `Mandate loaded in ${mandateLoadTime}ms`, 'execute-page');
        logger.info(`Starting auto-execution of mandate ${mandateId}`);

        // Set execution status in DOM for Playwright to detect
        const statusEl = document.createElement('div');
        statusEl.id = 'execution-status';
        statusEl.setAttribute('data-execution-status', 'initializing');
        statusEl.style.display = 'none';
        document.body.appendChild(statusEl);

        // Fetch API keys and provider settings
        setCurrentPhase('loading_config');
        eventEmitter.emitLog('info', 'Fetching API keys and provider settings', 'execute-page');
        const configStartTime = Date.now();
        
        const apiKeysResponse = await fetch('/api/export-api-keys');
        const apiKeys: Record<string, string> = apiKeysResponse.ok 
          ? await apiKeysResponse.json() 
          : {};
        
        const apiKeysCount = Object.keys(apiKeys).length;
        const apiKeysAvailable = Object.keys(apiKeys);
        eventEmitter.emitApiKeysLoaded(apiKeysCount, apiKeysAvailable, {
          load_time: Date.now() - configStartTime,
        });
        eventEmitter.emitLog('info', `Loaded ${apiKeysCount} API key(s): ${apiKeysAvailable.join(', ') || 'none'}`, 'execute-page');

        // Fetch provider settings
        const providersResponse = await fetch('/api/configured-providers');
        const providersData = providersResponse.ok 
          ? await providersResponse.json() as { providers?: Array<{ name: string; [key: string]: any }> }
          : { providers: [] };
        
        const providerSettings: Record<string, IProviderSetting> = {};
        if (providersData.providers) {
          for (const provider of providersData.providers) {
            providerSettings[provider.name] = provider as IProviderSetting;
            const model = provider.model || provider.defaultModel || 'unknown';
            eventEmitter.emitProviderConfigured(provider.name, model, {
              provider_config: provider,
            });
            eventEmitter.emitLog('info', `Provider configured: ${provider.name} (model: ${model})`, 'execute-page');
          }
        } else {
          eventEmitter.emitLog('warn', 'No provider settings found, using defaults', 'execute-page');
        }

        // Initialize WebContainer
        setCurrentPhase('initializing_webcontainer');
        eventEmitter.emitWebContainerInit('Starting WebContainer boot...', 0);
        eventEmitter.emitLog('info', 'Initializing WebContainer...', 'execute-page');
        const wcStartTime = Date.now();
        
        eventEmitter.emitWebContainerInit('Loading WebContainer kernel...', 25);
        const wc = await webcontainer;
        
        const wcInitTime = Date.now() - wcStartTime;
        eventEmitter.emitWebContainerInit('WebContainer ready', 100, {
          init_time: wcInitTime,
        });
        eventEmitter.emitLog('info', `WebContainer initialized in ${wcInitTime}ms`, 'execute-page');
        logger.info('WebContainer initialized');

        // Create shell terminal
        setCurrentPhase('initializing_shell');
        eventEmitter.emitLog('info', 'Creating shell terminal...', 'execute-page');
        const shellStartTime = Date.now();
        const shellTerminal = () => newBoltShellProcess();
        const shellInitTime = Date.now() - shellStartTime;
        eventEmitter.emitShellReady({ init_time: shellInitTime });
        eventEmitter.emitLog('info', `Shell terminal ready in ${shellInitTime}ms`, 'execute-page');

        // Create MandateExecutor
        setCurrentPhase('initializing_executor');
        eventEmitter.emitLog('info', 'Initializing MandateExecutor...', 'execute-page');
        const executorStartTime = Date.now();
        const executor = new MandateExecutor(
          mandate,
          Promise.resolve(wc),
          shellTerminal,
          apiKeys,
          providerSettings
        );
        const executorInitTime = Date.now() - executorStartTime;
        eventEmitter.emitExecutorReady({ init_time: executorInitTime });
        eventEmitter.emitLog('info', `MandateExecutor ready in ${executorInitTime}ms`, 'execute-page');
        
        const totalInitTime = Date.now() - initStartTime;
        eventEmitter.emitLog('info', `Initialization complete in ${totalInitTime}ms`, 'execute-page');

        // Set up event forwarding to governor and UI updates
        const executorEventEmitter = (executor as any).eventEmitter;
        if (executorEventEmitter) {
          executorEventEmitter.on('*', (event: any) => {
            // Update UI with recent logs
            if (event.type === 'log' || event.type === 'error') {
              setRecentLogs((prev) => {
                const newLogs = [...prev, {
                  level: event.data.level || (event.type === 'error' ? 'error' : 'info'),
                  message: event.data.message || '',
                  timestamp: event.timestamp,
                }];
                // Keep only last 10 logs
                return newLogs.slice(-10);
              });
            }
            
            // Update phase based on event type
            if (event.type === 'iteration_start') {
              setCurrentPhase(`executing_iteration_${event.iteration}`);
            } else if (event.type === 'iteration_end') {
              setCurrentPhase(`iteration_${event.iteration}_${event.data.status}`);
            }
            
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
        setCurrentPhase('executing');
        statusEl.setAttribute('data-execution-status', 'executing');

        // Execute mandate
        eventEmitter.emitLog('info', 'Starting mandate execution...', 'execute-page');
        const result = await executor.execute();

        // Store result
        resultRef.current = result;
        if (typeof window !== 'undefined') {
          (window as any).__EXECUTION_RESULT__ = result;
        }

        // Update status
        setCurrentPhase('completed');
        statusEl.setAttribute('data-execution-status', 'completed');

        // Log completion
        eventEmitter.emitLog('info', `Mandate execution completed with status: ${result.status}`, 'execute-page');
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
      <div className="flex-1 flex flex-col p-8">
        <div className="max-w-4xl mx-auto w-full">
          <h1 className="text-2xl font-bold text-bolt-elements-textPrimary mb-4">
            Executing Mandate
          </h1>
          <p className="text-bolt-elements-textSecondary mb-6">
            Mandate ID: <span className="font-mono">{mandateId}</span>
          </p>
          
          {/* Current Phase */}
          <div className="mb-6 p-4 bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg">
            <div className="flex items-center gap-3 mb-2">
              <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-accent-500"></div>
              <span className="text-sm font-medium text-bolt-elements-textSecondary">Current Phase:</span>
              <span className="text-sm font-semibold text-bolt-elements-textPrimary">{currentPhase}</span>
            </div>
            <p className="text-xs text-bolt-elements-textTertiary">
              Execution status is being reported to the governor and observability dashboard.
            </p>
          </div>

          {/* Recent Logs */}
          {recentLogs.length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-bolt-elements-textPrimary mb-2">Recent Logs</h2>
              <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-4 max-h-64 overflow-auto">
                <div className="space-y-1 font-mono text-xs">
                  {recentLogs.map((log, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className={`${
                        log.level === 'error' ? 'text-red-500' :
                        log.level === 'warn' ? 'text-yellow-500' :
                        log.level === 'debug' ? 'text-gray-500' :
                        'text-blue-500'
                      }`}>
                        [{log.level.toUpperCase()}]
                      </span>
                      <span className="text-bolt-elements-textSecondary">{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Status Info */}
          <div className="text-center">
            <p className="text-sm text-bolt-elements-textTertiary">
              This page is executing the mandate automatically.
              <br />
              View detailed logs at: <a href={`/observability/${mandateId}`} className="text-accent-500 hover:underline">/observability/{mandateId}</a>
            </p>
          </div>
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

