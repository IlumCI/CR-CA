/**
 * Workflow preview route.
 * 
 * Shows complete workflow from proposal creation through deployment.
 */

import { json, type LoaderFunctionArgs, type MetaFunction } from '@remix-run/cloudflare';
import { useLoaderData, useParams } from '@remix-run/react';
import { useEffect, useState } from 'react';
import { ClientOnly } from 'remix-utils/client-only';
import { WorkflowTimeline } from '~/components/workflow/WorkflowTimeline';
import { Workbench } from '~/components/workbench/Workbench.client';
import { Header } from '~/components/header/Header';
import BackgroundRays from '~/components/ui/BackgroundRays';
import type { WorkflowStatus } from '~/routes/api.workflow-status.$proposalId';

export const meta: MetaFunction<typeof loader> = ({ data }) => {
  return [
    { title: `Workflow - ${data?.proposalId || 'Unknown'}` },
    { name: 'description', content: 'Complete workflow visualization from proposal to deployment' },
  ];
};

export async function loader({ params }: LoaderFunctionArgs) {
  const proposalId = params.proposalId;
  
  if (!proposalId) {
    throw new Response('Proposal ID is required', { status: 400 });
  }

  // Fetch initial workflow status
  try {
    const baseUrl = process.env.BOLT_DIY_URL || 'http://localhost:5173';
    const response = await fetch(`${baseUrl}/api/workflow-status/${proposalId}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (response.ok) {
      const workflowStatus = await response.json();
      return json({ proposalId, initialStatus: workflowStatus });
    }
  } catch (error) {
    console.error('Error loading workflow status:', error);
  }

  return json({ proposalId, initialStatus: null });
}

export default function WorkflowRoute() {
  const { proposalId, initialStatus } = useLoaderData<typeof loader>();
  const params = useParams();
  const finalProposalId = proposalId || params.proposalId;
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(initialStatus);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    if (!finalProposalId) return;

    // Set up EventSource for real-time updates
    const baseUrl = window.location.origin;
    const eventSource = new EventSource(`${baseUrl}/api/workflow-status/${finalProposalId}?stream=true`);

    eventSource.onopen = () => {
      setIsConnected(true);
      setLastUpdate(new Date());
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setWorkflowStatus(data);
        setLastUpdate(new Date());
        setIsConnected(true);
      } catch (error) {
        console.error('Error parsing workflow status update:', error);
      }
    };

    let pollInterval: NodeJS.Timeout | null = null;
    
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      setIsConnected(false);
      eventSource.close();
      
      // Fallback to polling if EventSource fails
      pollInterval = setInterval(async () => {
        try {
          const response = await fetch(`${baseUrl}/api/workflow-status/${finalProposalId}`);
          if (response.ok) {
            const data = await response.json();
            setWorkflowStatus(data);
            setLastUpdate(new Date());
            setIsConnected(true);
          }
        } catch (error) {
          console.error('Error polling workflow status:', error);
          setIsConnected(false);
        }
      }, 5000);
    };

    return () => {
      eventSource.close();
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [finalProposalId]);

  if (!finalProposalId) {
    return (
      <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-bolt-elements-textPrimary mb-4">Error</h1>
          <p className="text-bolt-elements-textSecondary">Proposal ID is required</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1">
      <BackgroundRays />
      <Header />
      <div className="flex-1 overflow-hidden flex">
        {/* Workbench - Code Editor */}
        <div className="flex-1 min-w-0">
          <ClientOnly fallback={<div className="h-full flex items-center justify-center text-bolt-elements-textSecondary">Loading workbench...</div>}>
            {() => <Workbench chatStarted={true} />}
          </ClientOnly>
        </div>
        
        {/* Workflow Timeline Sidebar */}
        <div className="w-96 border-l border-bolt-elements-borderColor bg-bolt-elements-background-depth-2 overflow-auto">
          <div className="p-6">
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <h1 className="text-2xl font-bold text-bolt-elements-textPrimary">
                  Workflow Status
                </h1>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} title={isConnected ? 'Connected' : 'Disconnected'} />
                  <span className="text-xs text-bolt-elements-textSecondary">
                    {isConnected ? 'Live' : 'Polling'} • {lastUpdate.toLocaleTimeString()}
                  </span>
                </div>
              </div>
              <p className="text-sm text-bolt-elements-textSecondary">
                Proposal ID: {finalProposalId}
              </p>
            </div>
            
            {workflowStatus ? (
              <WorkflowTimeline workflowStatus={workflowStatus} />
            ) : (
              <div className="bg-bolt-elements-background-depth-3 border border-bolt-elements-borderColor rounded-lg p-8 text-center">
                <p className="text-bolt-elements-textSecondary">
                  Loading workflow status...
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

