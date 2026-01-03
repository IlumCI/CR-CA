import type { ExecutionEvent } from '~/types/mandate';
import { classNames } from '~/utils/classNames';

interface EventTimelineProps {
  events: ExecutionEvent[];
}

/**
 * EventTimeline component displays execution events in chronological order.
 */
export function EventTimeline({ events }: EventTimelineProps) {
  const getEventIcon = (type: ExecutionEvent['type']) => {
    switch (type) {
      case 'iteration_start':
        return 'i-ph:play-circle';
      case 'iteration_end':
        return 'i-ph:check-circle';
      case 'log':
        return 'i-ph:info';
      case 'error':
        return 'i-ph:x-circle';
      case 'diff':
        return 'i-ph:git-diff';
      case 'governance_check':
        return 'i-ph:shield-check';
      case 'deployment_status':
      case 'deployment_start':
      case 'deployment_end':
        return 'i-ph:rocket';
      default:
        return 'i-ph:circle';
    }
  };

  const getEventColor = (type: ExecutionEvent['type']) => {
    switch (type) {
      case 'iteration_start':
        return 'text-blue-500';
      case 'iteration_end':
        return 'text-green-500';
      case 'log':
        return 'text-gray-500';
      case 'error':
        return 'text-red-500';
      case 'diff':
        return 'text-purple-500';
      case 'governance_check':
        return 'text-yellow-500';
      case 'deployment_status':
      case 'deployment_start':
      case 'deployment_end':
        return 'text-indigo-500';
      default:
        return 'text-gray-400';
    }
  };

  if (events.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">No events yet. Waiting for execution to start...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {events.map((event, index) => (
        <div
          key={`${event.timestamp}-${index}`}
          className={classNames(
            'flex gap-4 p-4 rounded-lg border',
            'bg-bolt-elements-background-depth-3',
            'border-bolt-elements-borderColor'
          )}
        >
          <div className="flex-shrink-0">
            <div className={classNames('text-2xl', getEventColor(event.type))}>
              <span className={getEventIcon(event.type)} />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium text-bolt-elements-textPrimary capitalize">
                {event.type.replace(/_/g, ' ')}
              </span>
              <span className="text-xs text-bolt-elements-textTertiary">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {event.data.message && (
              <p className="text-sm text-bolt-elements-textSecondary mb-2">{event.data.message}</p>
            )}
            {event.data.status && (
              <span
                className={classNames(
                  'inline-block px-2 py-1 rounded text-xs font-medium',
                  event.data.status === 'success'
                    ? 'bg-green-500/20 text-green-500'
                    : event.data.status === 'failed'
                      ? 'bg-red-500/20 text-red-500'
                      : 'bg-blue-500/20 text-blue-500'
                )}
              >
                {event.data.status}
              </span>
            )}
            {event.iteration > 0 && (
              <div className="mt-2 text-xs text-bolt-elements-textTertiary">
                Iteration {event.iteration}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

