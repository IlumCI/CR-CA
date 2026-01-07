/**
 * Workflow status API endpoint.
 * 
 * Aggregates data from CorporateSwarm and bolt.diy to provide
 * complete workflow status from proposal to deployment.
 */

import { type LoaderFunctionArgs, json } from '@remix-run/cloudflare';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('api.workflow-status');

// Track if we've logged CorporateSwarm unavailability to prevent log spam
let corporateSwarmUnavailableLogged = false;

export interface WorkflowPhase {
  id: string;
  name: string;
  status: "pending" | "running" | "success" | "failed" | "skipped";
  timestamp?: number;
  duration?: number;
  details?: Record<string, any>;
  error?: string;
}

export interface WorkflowStatus {
  proposal_id: string;
  mandate_id?: string;
  phases: WorkflowPhase[];
  current_phase: string;
  overall_status: "pending" | "running" | "success" | "failed";
  created_at: number;
  updated_at: number;
  deployment_url?: string;
  test_results?: {
    status: string;
    tests_passed?: boolean;
    coverage?: number;
  };
}

export async function loader({ params, request }: LoaderFunctionArgs) {
  const proposalId = params.proposalId;
  
  if (!proposalId) {
    return json({ error: "proposal_id is required" }, { status: 400 });
  }

  // Try to get CorporateSwarm status (optional - CorporateSwarm may not be running)
  const corporateSwarmUrl = process.env.CORPORATE_SWARM_URL || "http://localhost:8000";
  let proposalData = null;
  let mandateData = null;
  
  try {
    const response = await fetch(`${corporateSwarmUrl}/api/proposal/${proposalId}`, {
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(2000) // Shorter timeout to fail fast
    });
    
    if (response.ok) {
      proposalData = await response.json();
    }
  } catch (e) {
    // CorporateSwarm is optional - only log once per server restart to reduce spam
    // The workflow can still function without CorporateSwarm data
    if (!corporateSwarmUnavailableLogged) {
      logger.debug(`CorporateSwarm not available at ${corporateSwarmUrl} (this is optional - workflow will continue without it)`);
      corporateSwarmUnavailableLogged = true;
    }
    // Silently continue - CorporateSwarm is optional
  }

  try {
    // Get mandate execution data from bolt.diy
    const mandateId = proposalData?.mandate_id || proposalData?.related_mandates?.[0];
    let executionData = null;

    if (mandateId) {
      try {
        // Get execution events from observability system
        const eventsResponse = await fetch(`${request.url.split('/api')[0]}/api/mandate/${mandateId}/events`, {
          headers: { "Content-Type": "application/json" },
          signal: AbortSignal.timeout(5000)
        });
        
        if (eventsResponse.ok) {
          executionData = await eventsResponse.json();
        }
      } catch (e) {
        logger.warn(`Could not fetch execution data: ${e}`);
      }
    }

    // Build workflow phases
    const phases: WorkflowPhase[] = [];

    // Phase 1: Proposal Creation
    phases.push({
      id: "proposal_creation",
      name: "Proposal Creation",
      status: proposalData ? "success" : "pending",
      timestamp: proposalData?.created_at,
      details: proposalData ? {
        title: proposalData.title,
        description: proposalData.description,
        sponsor: proposalData.sponsor
      } : undefined
    });

    // Phase 2: Governance Review
    if (proposalData?.vote_result) {
      phases.push({
        id: "governance_review",
        name: "Governance Review",
        status: proposalData.vote_result === "approved" ? "success" : "failed",
        timestamp: proposalData.vote_timestamp,
        details: {
          vote_result: proposalData.vote_result,
          participants: proposalData.vote_participants,
          consensus: proposalData.governance_consensus
        }
      });
    } else {
      phases.push({
        id: "governance_review",
        name: "Governance Review",
        status: "pending"
      });
    }

    // Phase 3: Mandate Creation
    if (mandateId) {
      phases.push({
        id: "mandate_creation",
        name: "Mandate Creation",
        status: "success",
        timestamp: executionData?.created_at,
        details: {
          mandate_id: mandateId,
          objectives: executionData?.objectives
        }
      });
    } else {
      phases.push({
        id: "mandate_creation",
        name: "Mandate Creation",
        status: "pending"
      });
    }

    // Phase 4: Code Generation
    if (executionData) {
      const codeGenEvents = executionData.events?.filter((e: any) => 
        e.type === "iteration_start" || e.type === "iteration_end"
      ) || [];
      
      if (codeGenEvents.length > 0) {
        const lastIteration = codeGenEvents[codeGenEvents.length - 1];
        phases.push({
          id: "code_generation",
          name: "Code Generation",
          status: lastIteration.data?.status === "success" ? "success" : 
                 lastIteration.data?.status === "failed" ? "failed" : "running",
          timestamp: lastIteration.timestamp,
          details: {
            iterations: codeGenEvents.length,
            files_created: executionData.final_state?.files_created || [],
            files_modified: executionData.final_state?.files_modified || []
          }
        });
      } else {
        phases.push({
          id: "code_generation",
          name: "Code Generation",
          status: "running"
        });
      }
    } else {
      phases.push({
        id: "code_generation",
        name: "Code Generation",
        status: "pending"
      });
    }

    // Phase 5: Testing
    if (executionData?.test_results) {
      phases.push({
        id: "testing",
        name: "Testing",
        status: executionData.test_results.tests_passed ? "success" : "failed",
        timestamp: executionData.test_results.timestamp,
        details: {
          tests_passed: executionData.test_results.tests_passed,
          coverage: executionData.test_results.coverage,
          test_files: executionData.test_results.test_files
        }
      });
    } else if (executionData?.final_state?.build_successful) {
      phases.push({
        id: "testing",
        name: "Testing",
        status: "pending"
      });
    }

    // Phase 6: Git Operations
    if (executionData?.io_results?.git_operations) {
      const gitOps = executionData.io_results.git_operations;
      const gitStatus = gitOps.status === "completed" || gitOps.status === "success" 
        ? "success" 
        : gitOps.status === "failed" || gitOps.status === "error"
        ? "failed"
        : "pending";
      
      phases.push({
        id: "git_operations",
        name: "Git Operations",
        status: gitStatus,
        details: gitOps
      });
    }

    // Phase 7: Deployment
    if (executionData?.deployment) {
      const deployStatus = executionData.deployment.status;
      phases.push({
        id: "deployment",
        name: "Deployment",
        status: deployStatus === "success" ? "success" : 
               deployStatus === "failed" ? "failed" : "running",
        timestamp: executionData.deployment.timestamp,
        details: {
          platform: executionData.deployment.platform,
          url: executionData.deployment.url,
          deployment_id: executionData.deployment.deployment_id
        },
        error: executionData.deployment.error
      });
    } else if (executionData?.final_state?.build_successful) {
      phases.push({
        id: "deployment",
        name: "Deployment",
        status: "pending"
      });
    }

    // Determine overall status
    const failedPhases = phases.filter(p => p.status === "failed");
    const runningPhases = phases.filter(p => p.status === "running");
    const successPhases = phases.filter(p => p.status === "success");
    
    let overall_status: "pending" | "running" | "success" | "failed" = "pending";
    if (failedPhases.length > 0) {
      overall_status = "failed";
    } else if (runningPhases.length > 0) {
      overall_status = "running";
    } else if (successPhases.length === phases.length && phases.length > 0) {
      overall_status = "success";
    }

    // Find current phase
    const currentPhase = phases.find(p => p.status === "running") || 
                        phases.find(p => p.status === "pending") ||
                        phases[phases.length - 1];

    const workflowStatus: WorkflowStatus = {
      proposal_id: proposalId,
      mandate_id: mandateId,
      phases,
      current_phase: currentPhase?.id || "proposal_creation",
      overall_status,
      created_at: proposalData?.created_at || Date.now() / 1000,
      updated_at: Date.now() / 1000,
      deployment_url: executionData?.deployment?.url,
      test_results: executionData?.test_results ? {
        status: executionData.test_results.status,
        tests_passed: executionData.test_results.tests_passed,
        coverage: executionData.test_results.coverage
      } : undefined
    };

    return json(workflowStatus);
  } catch (error) {
    // Only log actual errors, not expected CorporateSwarm unavailability
    if (error instanceof Error && !error.message.includes('fetch failed') && !error.message.includes('timeout')) {
      logger.error(`Error fetching workflow status: ${error}`);
    }
    
    // Return a minimal workflow status even if CorporateSwarm is unavailable
    const fallbackStatus: WorkflowStatus = {
      proposal_id: proposalId,
      phases: [
        {
          id: "proposal_creation",
          name: "Proposal Creation",
          status: "pending"
        }
      ],
      current_phase: "proposal_creation",
      overall_status: "pending",
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000
    };
    
    return json(fallbackStatus);
  }
}

