import { json, type MetaFunction, type LoaderFunctionArgs } from '@remix-run/cloudflare';
import { useLoaderData, Link } from '@remix-run/react';
import { useEffect, useState } from 'react';
import { Header } from '~/components/header/Header';
import BackgroundRays from '~/components/ui/BackgroundRays';
import { MandateSubmission } from '~/components/mandate/MandateSubmission';
import { eventRegistry } from '~/lib/runtime/execution-events';

export const meta: MetaFunction = () => {
  return [{ title: 'Bolt.diy - LLM-Native Execution Engine' }, { name: 'description', content: 'Autonomous code execution via structured mandates' }];
};

/**
 * Load recent mandates from event registry.
 */
export async function loader({ request }: LoaderFunctionArgs) {
  try {
    // Get all active mandate IDs from event registry
    const allMandateIds = eventRegistry.getActiveMandates();
    
    // Get recent mandates (last 10)
    const recentMandates = allMandateIds.slice(-10).reverse().map(mandateId => {
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
    
    return json({ recentMandates });
  } catch (error) {
    console.error('Error loading recent mandates:', error);
    return json({ recentMandates: [] });
  }
}

/**
 * Landing page component for Bolt.diy
 * LLM-native execution engine: accepts structured mandates from CorporateSwarm
 * and executes code generation autonomously with full governance oversight.
 */
export default function Index() {
  const { recentMandates: initialMandates } = useLoaderData<typeof loader>();
  const [recentMandates, setRecentMandates] = useState(initialMandates);
  const [isLive, setIsLive] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Real-time updates via polling
  useEffect(() => {
    let pollInterval: NodeJS.Timeout | null = null;
    let isMounted = true;

    const updateMandates = async () => {
      try {
        // Fetch active mandates from API endpoint
        const response = await fetch('/api/mandate?list=true');
        if (response.ok) {
          const data = await response.json() as { mandates?: typeof recentMandates };
          const updatedMandates = data.mandates || [];

          if (isMounted) {
            setRecentMandates(updatedMandates);
            setLastUpdate(new Date());
            setIsLive(true);
          }
        } else {
          if (isMounted) {
            setIsLive(false);
          }
        }
      } catch (error) {
        console.error('Error updating mandates:', error);
        if (isMounted) {
          setIsLive(false);
        }
      }
    };

    // Initial update
    updateMandates();

    // Poll every 2 seconds
    pollInterval = setInterval(updateMandates, 2000);

    return () => {
      isMounted = false;
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, []);
  
  return (
    <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1">
      <BackgroundRays />
      <Header />
      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-7xl mx-auto w-full space-y-8">
          {/* Recent Mandates Section */}
          {recentMandates && recentMandates.length > 0 && (
            <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-bolt-elements-textPrimary">
                  Recent Mandates
                </h2>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-red-500'}`} title={isLive ? 'Live' : 'Disconnected'} />
                  <span className="text-xs text-bolt-elements-textSecondary">
                    {isLive ? 'Live' : 'Disconnected'} • Updated {lastUpdate.toLocaleTimeString()}
                  </span>
                </div>
              </div>
              <div className="space-y-3">
                {recentMandates.map((mandate) => (
                  <div
                    key={mandate.mandate_id}
                    className="flex items-center justify-between p-4 bg-bolt-elements-background-depth-3 border border-bolt-elements-borderColor rounded-md hover:border-accent-500/50 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-mono text-bolt-elements-textSecondary">
                          {mandate.mandate_id.substring(0, 8)}...
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          mandate.status === 'completed' ? 'bg-green-500/20 text-green-500' :
                          mandate.status === 'failed' ? 'bg-red-500/20 text-red-500' :
                          mandate.status === 'running' ? 'bg-blue-500/20 text-blue-500' :
                          'bg-gray-500/20 text-gray-500'
                        }`}>
                          {mandate.status}
                        </span>
                        <span className="text-xs text-bolt-elements-textTertiary">
                          {mandate.event_count} events
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Link
                        to={`/observability/${mandate.mandate_id}`}
                        className="px-3 py-1 text-sm bg-accent-500/10 text-accent-500 rounded hover:bg-accent-500/20 transition-colors"
                      >
                        View
                      </Link>
                      <Link
                        to={`/execute/${mandate.mandate_id}`}
                        className="px-3 py-1 text-sm bg-blue-500/10 text-blue-500 rounded hover:bg-blue-500/20 transition-colors"
                      >
                        Execute
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Mandate Submission Section */}
          <div className="bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg p-6">
            <h2 className="text-xl font-semibold text-bolt-elements-textPrimary mb-4">
              Submit New Mandate
            </h2>
            <MandateSubmission />
          </div>
        </div>
      </div>
    </div>
  );
}
