import { useState, useEffect, useCallback } from 'react';
import { EventTimeline } from './EventTimeline';
import { MandateDetails } from './MandateDetails';
import { FileDiffViewer } from './FileDiffViewer';
import { LogStream } from './LogStream';
import { GovernanceStatus } from './GovernanceStatus';
import { DeploymentStatus } from './DeploymentStatus';
import { GovernorMetrics } from './GovernorMetrics';
import type { ExecutionEvent } from '~/types/mandate';
import { createScopedLogger } from '~/utils/logger';
import { classNames } from '~/utils/classNames';

const logger = createScopedLogger('ObservabilityDashboard');

interface ObservabilityDashboardProps {
  mandateId: string;
}

type ViewMode = 'timeline' | 'details' | 'diffs' | 'logs' | 'governance' | 'deployment' | 'governor';

/**
 * Main observability dashboard component.
 * Displays real-time execution state, diffs, logs, and governance metadata.
 */
export function ObservabilityDashboard({ mandateId }: ObservabilityDashboardProps) {
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [currentView, setCurrentView] = useState<ViewMode>('timeline');
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Connect to event stream
  useEffect(() => {
    let eventSource: EventSource | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connect = () => {
      try {
        const streamUrl = `/api/mandate?stream=true&mandate_id=${mandateId}`;
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = () => {
          logger.info(`Connected to event stream for mandate ${mandateId}`);
          setIsConnected(true);
          setError(null);
          reconnectAttempts = 0;
        };

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Handle connection confirmation
            if (data.type === 'connected') {
              logger.debug('Event stream connected');
              return;
            }

            // Add new event
            setEvents((prev) => {
              // Avoid duplicates
              const exists = prev.some((e) => e.timestamp === data.timestamp && e.type === data.type);
              if (exists) return prev;
              return [...prev, data].sort((a, b) => a.timestamp - b.timestamp);
            });
          } catch (err) {
            logger.error('Error parsing event:', err);
          }
        };

        eventSource.onerror = (err) => {
          logger.error('Event stream error:', err);
          setIsConnected(false);
          
          if (eventSource?.readyState === EventSource.CLOSED) {
            eventSource.close();
            
            // Attempt reconnection
            if (reconnectAttempts < maxReconnectAttempts) {
              reconnectAttempts++;
              logger.info(`Reconnecting... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
              setTimeout(connect, 1000 * reconnectAttempts);
            } else {
              setError('Failed to connect to event stream after multiple attempts');
            }
          }
        };
      } catch (err) {
        logger.error('Error setting up event stream:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    connect();

    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [mandateId]);

  // Load initial events via polling (fallback)
  useEffect(() => {
    const loadInitialEvents = async () => {
      try {
        const response = await fetch(`/api/mandate?mandate_id=${mandateId}`);
        if (response.ok) {
          const data = await response.json() as { events?: ExecutionEvent[] };
          if (data.events && Array.isArray(data.events)) {
            setEvents(data.events.sort((a: ExecutionEvent, b: ExecutionEvent) => a.timestamp - b.timestamp));
          }
        }
      } catch (err) {
        logger.error('Error loading initial events:', err);
      }
    };

    // Only load if we don't have events yet
    if (events.length === 0) {
      loadInitialEvents();
    }
  }, [mandateId, events.length]);

  const viewButtons: { mode: ViewMode; label: string; icon: string }[] = [
    { mode: 'timeline', label: 'Timeline', icon: 'i-ph:clock-clockwise' },
    { mode: 'details', label: 'Details', icon: 'i-ph:file-text' },
    { mode: 'diffs', label: 'Diffs', icon: 'i-ph:git-diff' },
    { mode: 'logs', label: 'Logs', icon: 'i-ph:terminal' },
    { mode: 'governance', label: 'Governance', icon: 'i-ph:shield-check' },
    { mode: 'deployment', label: 'Deployment', icon: 'i-ph:rocket' },
    { mode: 'governor', label: 'Governor', icon: 'i-ph:gear' },
  ];

  return (
    <div className="flex flex-col h-full w-full p-8">
      <div className="max-w-7xl mx-auto w-full">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-2xl font-bold text-bolt-elements-textPrimary">
              Execution Observability
            </h1>
            <div className="flex items-center gap-2">
              <div
                className={classNames(
                  'w-2 h-2 rounded-full',
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                )}
                title={isConnected ? 'Connected' : 'Disconnected'}
              />
              <span className="text-sm text-bolt-elements-textSecondary">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
          </div>
          <p className="text-sm text-bolt-elements-textSecondary">
            Mandate ID: <span className="font-mono">{mandateId}</span>
          </p>
          {error && (
            <div className="mt-2 p-2 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-500">
              {error}
            </div>
          )}
        </div>

        {/* View Tabs */}
        <div className="mb-4 flex gap-2 border-b border-bolt-elements-borderColor">
          {viewButtons.map(({ mode, label, icon }) => (
            <button
              key={mode}
              onClick={() => setCurrentView(mode)}
              className={classNames(
                'px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px',
                currentView === mode
                  ? 'text-accent-500 border-accent-500'
                  : 'text-bolt-elements-textSecondary border-transparent hover:text-bolt-elements-textPrimary'
              )}
            >
              <span className={classNames(icon, 'mr-2')} />
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6 min-h-[600px]">
          {currentView === 'timeline' && <EventTimeline events={events} />}
          {currentView === 'details' && <MandateDetails mandateId={mandateId} events={events} />}
          {currentView === 'diffs' && <FileDiffViewer events={events} />}
          {currentView === 'logs' && <LogStream events={events} />}
          {currentView === 'governance' && <GovernanceStatus events={events} />}
          {currentView === 'deployment' && <DeploymentStatus events={events} />}
          {currentView === 'governor' && <GovernorMetrics mandateId={mandateId} />}
        </div>
      </div>
    </div>
  );
}

