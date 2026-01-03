import { useState, useMemo } from 'react';
import type { ExecutionEvent } from '~/types/mandate';
import { classNames } from '~/utils/classNames';

interface FileDiffViewerProps {
  events: ExecutionEvent[];
}

/**
 * FileDiffViewer component displays file changes per iteration.
 */
export function FileDiffViewer({ events }: FileDiffViewerProps) {
  const [selectedIteration, setSelectedIteration] = useState<number | null>(null);

  const diffEvents = useMemo(() => {
    return events.filter((event) => event.type === 'diff');
  }, [events]);

  const iterations = useMemo(() => {
    const uniqueIterations = new Set(diffEvents.map((e) => e.iteration));
    return Array.from(uniqueIterations).sort((a, b) => a - b);
  }, [diffEvents]);

  const selectedDiffs = useMemo(() => {
    if (selectedIteration === null) {
      return diffEvents;
    }
    return diffEvents.filter((e) => e.iteration === selectedIteration);
  }, [diffEvents, selectedIteration]);

  if (diffEvents.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-bolt-elements-textSecondary">
          No file changes yet. Diffs will appear here as execution progresses.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Iteration Filter */}
      {iterations.length > 1 && (
        <div className="mb-4 flex gap-2 flex-wrap">
          <button
            onClick={() => setSelectedIteration(null)}
            className={classNames(
              'px-3 py-1 rounded text-sm font-medium transition-colors',
              selectedIteration === null
                ? 'bg-accent-500 text-white'
                : 'bg-bolt-elements-background-depth-3 text-bolt-elements-textSecondary hover:text-bolt-elements-textPrimary'
            )}
          >
            All Iterations
          </button>
          {iterations.map((iter) => (
            <button
              key={iter}
              onClick={() => setSelectedIteration(iter)}
              className={classNames(
                'px-3 py-1 rounded text-sm font-medium transition-colors',
                selectedIteration === iter
                  ? 'bg-accent-500 text-white'
                  : 'bg-bolt-elements-background-depth-3 text-bolt-elements-textSecondary hover:text-bolt-elements-textPrimary'
              )}
            >
              Iteration {iter}
            </button>
          ))}
        </div>
      )}

      {/* Diffs */}
      <div className="flex-1 overflow-auto space-y-4">
        {selectedDiffs.map((event, index) => {
          const filesChanged = event.data.files_changed || [];
          const timestamp = new Date(event.timestamp).toLocaleString();

          return (
            <div
              key={`${event.timestamp}-${index}`}
              className="p-4 rounded-lg border border-bolt-elements-borderColor bg-bolt-elements-background-depth-3"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="i-ph:git-diff text-purple-500" />
                  <span className="font-medium text-bolt-elements-textPrimary">
                    Iteration {event.iteration}
                  </span>
                </div>
                <span className="text-xs text-bolt-elements-textTertiary">{timestamp}</span>
              </div>

              {filesChanged.length > 0 ? (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-bolt-elements-textSecondary mb-2">
                    Files Changed ({filesChanged.length}):
                  </div>
                  <ul className="space-y-1">
                    {filesChanged.map((file, fileIdx) => (
                      <li
                        key={fileIdx}
                        className="text-sm font-mono text-bolt-elements-textPrimary bg-bolt-elements-background-depth-4 p-2 rounded"
                      >
                        {file}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-sm text-bolt-elements-textSecondary">No file changes in this iteration.</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

