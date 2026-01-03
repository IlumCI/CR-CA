import { json, type LoaderFunctionArgs, type MetaFunction } from '@remix-run/cloudflare';
import { useLoaderData, useParams } from '@remix-run/react';
import { ObservabilityDashboard } from '~/components/observability/ObservabilityDashboard';
import { Header } from '~/components/header/Header';
import BackgroundRays from '~/components/ui/BackgroundRays';

export const meta: MetaFunction<typeof loader> = ({ data }) => {
  return [
    { title: `Observability - ${data?.mandateId || 'Unknown'}` },
    { name: 'description', content: 'Real-time execution observability for mandate execution' },
  ];
};

export async function loader({ params }: LoaderFunctionArgs) {
  const mandateId = params.mandateId;
  
  if (!mandateId) {
    throw new Response('Mandate ID is required', { status: 400 });
  }

  return json({ mandateId });
}

export default function ObservabilityRoute() {
  const { mandateId } = useLoaderData<typeof loader>();
  const params = useParams();

  // Use params as fallback if loader data is not available
  const finalMandateId = mandateId || params.mandateId;

  if (!finalMandateId) {
    return (
      <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-bolt-elements-textPrimary mb-4">Error</h1>
          <p className="text-bolt-elements-textSecondary">Mandate ID is required</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1">
      <BackgroundRays />
      <Header />
      <ObservabilityDashboard mandateId={finalMandateId} />
    </div>
  );
}

