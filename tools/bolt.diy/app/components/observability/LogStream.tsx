import { useState, useMemo } from 'react';
import type { ExecutionEvent } from '~/types/mandate';
import { classNames } from '~/utils/classNames';

interface LogStreamProps {
  events: ExecutionEvent[];
}

type LogLevel = 'all' | 'info' | 'warn' | 'error' | 'debug';

/**
 * LogStream component displays real-time logs from execution events.
 */
export function LogStream({ events }: LogStreamProps) {
  const [logLevel, setLogLevel] = useState<LogLevel>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const logEvents = useMemo(() => {
    // Include all event types that have log-like data
    return events.filter((event) => 
      event.type === 'log' || 
      event.type === 'error' ||
      event.type === 'initialization_start' ||
      event.type === 'webcontainer_init' ||
      event.type === 'api_keys_loaded' ||
      event.type === 'provider_configured' ||
      event.type === 'shell_ready' ||
      event.type === 'executor_ready' ||
      event.type === 'iteration_start' ||
      event.type === 'iteration_end' ||
      event.type === 'governance_check' ||
      event.type === 'deployment_status' ||
      event.type === 'budget_warning' ||
      event.type === 'constraint_violation'
    );
  }, [events]);

  const filteredLogs = useMemo(() => {
    let filtered = logEvents;

    // Filter by level
    if (logLevel !== 'all') {
      filtered = filtered.filter((event) => {
        if (event.type === 'error') return logLevel === 'error';
        if (event.type === 'constraint_violation') return logLevel === 'error';
        if (event.type === 'budget_warning') return logLevel === 'warn';
        // For initialization events, treat as info unless they're errors
        if (['initialization_start', 'webcontainer_init', 'api_keys_loaded', 'provider_configured', 'shell_ready', 'executor_ready'].includes(event.type)) {
          return logLevel === 'info' || logLevel === 'debug';
        }
        return event.data.level === logLevel;
      });
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((event) => {
        const message = event.data.message?.toLowerCase() || '';
        const source = event.data.source?.toLowerCase() || '';
        const type = event.type.toLowerCase();
        return message.includes(query) || source.includes(query) || type.includes(query);
      });
    }

    return filtered;
  }, [logEvents, logLevel, searchQuery]);

  const logLevels: { value: LogLevel; label: string; color: string }[] = [
    { value: 'all', label: 'All', color: 'text-gray-500' },
    { value: 'info', label: 'Info', color: 'text-blue-500' },
    { value: 'warn', label: 'Warn', color: 'text-yellow-500' },
    { value: 'error', label: 'Error', color: 'text-red-500' },
    { value: 'debug', label: 'Debug', color: 'text-gray-400' },
  ];

  const getLogColor = (level?: string) => {
    switch (level) {
      case 'info':
        return 'text-blue-500';
      case 'warn':
        return 'text-yellow-500';
      case 'error':
        return 'text-red-500';
      case 'debug':
        return 'text-gray-400';
      default:
        return 'text-bolt-elements-textSecondary';
    }
  };

  if (logEvents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">No logs yet. Waiting for execution to start...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Filters */}
      <div className="mb-4 flex gap-4 items-center">
        <div className="flex gap-2">
          {logLevels.map(({ value, label, color }) => (
            <button
              key={value}
              onClick={() => setLogLevel(value)}
              className={classNames(
                'px-3 py-1 rounded text-sm font-medium transition-colors',
                logLevel === value
                  ? 'bg-accent-500 text-white'
                  : 'bg-bolt-elements-background-depth-3 text-bolt-elements-textSecondary hover:text-bolt-elements-textPrimary'
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search logs..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex-1 px-3 py-1 rounded bg-bolt-elements-background-depth-3 border border-bolt-elements-borderColor text-sm text-bolt-elements-textPrimary placeholder:text-bolt-elements-textTertiary"
        />
      </div>

      {/* Logs */}
      <div className="flex-1 overflow-auto font-mono text-sm space-y-1">
        {filteredLogs.length === 0 ? (
          <div className="text-center py-8 text-bolt-elements-textSecondary">
            No logs match the current filters.
          </div>
        ) : (
          filteredLogs.map((event, index) => {
            // Determine level and message based on event type
            let level: string = 'info';
            let message: string = '';
            let eventTypeLabel: string = event.type;
            
            if (event.type === 'error' || event.type === 'constraint_violation') {
              level = 'error';
              message = event.data.message || event.data.error_message || event.data.violation_details || 'Unknown error';
            } else if (event.type === 'budget_warning') {
              level = 'warn';
              message = event.data.message || 'Budget warning';
            } else if (event.type === 'log') {
              level = event.data.level || 'info';
              message = event.data.message || '';
            } else if (event.type === 'initialization_start') {
              level = 'info';
              message = event.data.message || 'Initialization started';
              eventTypeLabel = 'INIT';
            } else if (event.type === 'webcontainer_init') {
              level = 'debug';
              message = event.data.message || 'WebContainer initialization';
              eventTypeLabel = 'WC';
            } else if (event.type === 'api_keys_loaded') {
              level = 'info';
              message = event.data.message || `Loaded ${event.data.keys_count || 0} API key(s)`;
              eventTypeLabel = 'API';
            } else if (event.type === 'provider_configured') {
              level = 'info';
              message = event.data.message || `Provider configured: ${event.data.provider || 'unknown'}`;
              eventTypeLabel = 'PROV';
            } else if (event.type === 'shell_ready') {
              level = 'info';
              message = event.data.message || 'Shell terminal ready';
              eventTypeLabel = 'SHELL';
            } else if (event.type === 'executor_ready') {
              level = 'info';
              message = event.data.message || 'MandateExecutor ready';
              eventTypeLabel = 'EXEC';
            } else if (event.type === 'iteration_start') {
              level = 'info';
              message = `Iteration ${event.data.iteration_number || event.iteration} started`;
              eventTypeLabel = 'ITER';
            } else if (event.type === 'iteration_end') {
              level = event.data.status === 'failed' ? 'error' : 'info';
              message = `Iteration ${event.data.iteration_number || event.iteration} ${event.data.status || 'completed'}`;
              eventTypeLabel = 'ITER';
            } else if (event.type === 'governance_check') {
              level = 'info';
              message = 'Governance check performed';
              eventTypeLabel = 'GOV';
            } else if (event.type === 'deployment_status') {
              level = event.data.deployment_status === 'failed' ? 'error' : 'info';
              message = event.data.message || `Deployment ${event.data.deployment_status || 'unknown'}`;
              eventTypeLabel = 'DEPLOY';
            } else {
              message = event.data.message || JSON.stringify(event.data);
            }
            
            const timestamp = new Date(event.timestamp).toISOString();
            const source = event.data.source || eventTypeLabel;
            const timing = event.metadata?.init_time || event.metadata?.load_time || event.metadata?.execution_time;
            const timingStr = timing ? ` (${timing}ms)` : '';

            return (
              <div
                key={`${event.timestamp}-${index}`}
                className={classNames(
                  'p-2 rounded border-l-2',
                  level === 'error'
                    ? 'bg-red-500/10 border-red-500'
                    : level === 'warn'
                      ? 'bg-yellow-500/10 border-yellow-500'
                      : level === 'debug'
                        ? 'bg-gray-500/5 border-gray-500/30'
                        : 'bg-bolt-elements-background-depth-3 border-bolt-elements-borderColor'
                )}
              >
                <div className="flex gap-2 items-center flex-wrap">
                  <span className={classNames('font-semibold text-xs', getLogColor(level))}>
                    [{level.toUpperCase()}]
                  </span>
                  <span className="text-xs font-mono text-bolt-elements-textTertiary">{eventTypeLabel}</span>
                  <span className="text-xs text-bolt-elements-textTertiary">{timestamp}</span>
                  {source && source !== eventTypeLabel && (
                    <span className="text-xs text-bolt-elements-textTertiary">from {source}</span>
                  )}
                  {timingStr && (
                    <span className="text-xs text-bolt-elements-textTertiary">{timingStr}</span>
                  )}
                </div>
                <div className={classNames('mt-1 text-sm', getLogColor(level))}>{message}</div>
                {event.metadata && Object.keys(event.metadata).length > 0 && (
                  <details className="mt-1 text-xs text-bolt-elements-textTertiary">
                    <summary className="cursor-pointer hover:text-bolt-elements-textSecondary">Details</summary>
                    <pre className="mt-1 p-2 bg-bolt-elements-background-depth-2 rounded text-xs overflow-auto">
                      {JSON.stringify(event.metadata, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

