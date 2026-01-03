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
    return events.filter((event) => event.type === 'log' || event.type === 'error');
  }, [events]);

  const filteredLogs = useMemo(() => {
    let filtered = logEvents;

    // Filter by level
    if (logLevel !== 'all') {
      filtered = filtered.filter((event) => {
        if (event.type === 'error') return logLevel === 'error';
        return event.data.level === logLevel;
      });
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((event) => {
        const message = event.data.message?.toLowerCase() || '';
        return message.includes(query);
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
            const level = event.type === 'error' ? 'error' : event.data.level || 'info';
            const timestamp = new Date(event.timestamp).toISOString();
            const message = event.data.message || '';

            return (
              <div
                key={`${event.timestamp}-${index}`}
                className={classNames(
                  'p-2 rounded border-l-2',
                  level === 'error'
                    ? 'bg-red-500/10 border-red-500'
                    : level === 'warn'
                      ? 'bg-yellow-500/10 border-yellow-500'
                      : 'bg-bolt-elements-background-depth-3 border-bolt-elements-borderColor'
                )}
              >
                <div className="flex gap-2">
                  <span className={classNames('font-semibold', getLogColor(level))}>
                    [{level.toUpperCase()}]
                  </span>
                  <span className="text-bolt-elements-textTertiary">{timestamp}</span>
                </div>
                <div className={classNames('mt-1', getLogColor(level))}>{message}</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

