import { json, type MetaFunction } from '@remix-run/cloudflare';
import { Header } from '~/components/header/Header';
import BackgroundRays from '~/components/ui/BackgroundRays';
import { MandateSubmission } from '~/components/mandate/MandateSubmission';

export const meta: MetaFunction = () => {
  return [{ title: 'Bolt.diy - LLM-Native Execution Engine' }, { name: 'description', content: 'Autonomous code execution via structured mandates' }];
};

export const loader = () => json({});

/**
 * Landing page component for Bolt.diy
 * LLM-native execution engine: accepts structured mandates from CorporateSwarm
 * and executes code generation autonomously with full governance oversight.
 */
export default function Index() {
  return (
    <div className="flex flex-col h-full w-full bg-bolt-elements-background-depth-1">
      <BackgroundRays />
      <Header />
      <MandateSubmission />
    </div>
  );
}
