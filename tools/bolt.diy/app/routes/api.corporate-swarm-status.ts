/**
 * CorporateSwarm status API endpoint.
 * 
 * Queries CorporateSwarm for proposal and mandate status.
 */

import { type LoaderFunctionArgs, json } from '@remix-run/cloudflare';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('api.corporate-swarm-status');

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const proposalId = url.searchParams.get("proposal_id");
  const mandateId = url.searchParams.get("mandate_id");

  if (!proposalId && !mandateId) {
    return json({ error: "proposal_id or mandate_id is required" }, { status: 400 });
  }

  try {
    const corporateSwarmUrl = process.env.CORPORATE_SWARM_URL || "http://localhost:8000";
    let data = null;

    if (proposalId) {
      const response = await fetch(`${corporateSwarmUrl}/api/proposal/${proposalId}`, {
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        data = await response.json();
      } else {
        return json({ error: "Proposal not found" }, { status: 404 });
      }
    } else if (mandateId) {
      const response = await fetch(`${corporateSwarmUrl}/api/mandate/${mandateId}`, {
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        data = await response.json();
      } else {
        return json({ error: "Mandate not found" }, { status: 404 });
      }
    }

    return json(data || {});
  } catch (error) {
    logger.error(`Error fetching CorporateSwarm status: ${error}`);
    return json({ error: "Failed to fetch CorporateSwarm status" }, { status: 500 });
  }
}

