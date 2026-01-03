/**
 * HTTP client for CorporateSwarm integration.
 * 
 * Provides communication between bolt.diy and CorporateSwarm
 * for governance oversight, reporting, and approval workflows.
 */

import { createScopedLogger } from '~/utils/logger';
import type { ExecutionResult, ExecutionEvent } from '~/types/mandate';

const logger = createScopedLogger('corporate-swarm-client');

/**
 * Configuration for CorporateSwarm client.
 */
export interface CorporateSwarmConfig {
  baseUrl: string; // e.g., 'http://localhost:8000' or 'https://corposwarm.example.com'
  apiKey?: string;
  timeout?: number; // milliseconds
}

/**
 * Execution report for CorporateSwarm.
 */
export interface ExecutionReport {
  mandate_id: string;
  proposal_id?: string;
  status: ExecutionResult['status'];
  iterations_completed: number;
  governance_summary: ExecutionResult['governance_summary'];
  budget_summary: ExecutionResult['budget_summary'];
  deployment_result?: ExecutionResult['deployment_result'];
  errors?: ExecutionResult['errors'];
  events: ExecutionEvent[];
  created_at: number;
  completed_at: number;
}

/**
 * Approval request for CorporateSwarm.
 */
export interface ApprovalRequest {
  mandate_id: string;
  proposal_id?: string;
  checkpoint: string;
  context: {
    iteration?: number;
    risk_level?: string;
    budget_usage?: number;
    description: string;
  };
}

/**
 * Approval response from CorporateSwarm.
 */
export interface ApprovalResponse {
  approved: boolean;
  reason?: string;
  conditions?: string[];
  timestamp: number;
}

/**
 * HTTP client for CorporateSwarm integration.
 */
export class CorporateSwarmClient {
  private config: CorporateSwarmConfig;

  constructor(config: CorporateSwarmConfig) {
    this.config = {
      timeout: 30000, // 30 seconds default
      ...config,
    };
  }

  /**
   * Submit execution report to CorporateSwarm.
   */
  async submitExecutionReport(report: ExecutionReport): Promise<void> {
    try {
      const url = `${this.config.baseUrl}/api/execution/report`;
      const response = await this.fetchWithTimeout(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(report),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to submit execution report: ${response.status} ${errorText}`);
      }

      logger.info(`Execution report submitted for mandate ${report.mandate_id}`);
    } catch (error) {
      logger.error(`Error submitting execution report: ${error}`);
      throw error;
    }
  }

  /**
   * Request approval from CorporateSwarm at a checkpoint.
   */
  async requestApproval(request: ApprovalRequest): Promise<ApprovalResponse> {
    try {
      const url = `${this.config.baseUrl}/api/execution/approval`;
      const response = await this.fetchWithTimeout(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to request approval: ${response.status} ${errorText}`);
      }

      const approval: ApprovalResponse = await response.json();
      logger.info(
        `Approval ${approval.approved ? 'granted' : 'denied'} for mandate ${request.mandate_id} at checkpoint ${request.checkpoint}`
      );

      return approval;
    } catch (error) {
      logger.error(`Error requesting approval: ${error}`);
      // Default to approved if CorporateSwarm is unavailable (fail-open for development)
      // In production, this should be configurable
      return {
        approved: false,
        reason: `Error connecting to CorporateSwarm: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: Date.now(),
      };
    }
  }

  /**
   * Update proposal status in CorporateSwarm.
   */
  async updateProposalStatus(
    proposal_id: string,
    status: 'pending' | 'executing' | 'completed' | 'failed' | 'cancelled',
    details?: {
      mandate_id?: string;
      progress?: number;
      message?: string;
    }
  ): Promise<void> {
    try {
      const url = `${this.config.baseUrl}/api/proposals/${proposal_id}/status`;
      const response = await this.fetchWithTimeout(url, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify({
          status,
          ...details,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update proposal status: ${response.status} ${errorText}`);
      }

      logger.info(`Proposal ${proposal_id} status updated to ${status}`);
    } catch (error) {
      logger.error(`Error updating proposal status: ${error}`);
      throw error;
    }
  }

  /**
   * Get mandate configuration from CorporateSwarm.
   */
  async getMandateConfig(mandate_id: string): Promise<any> {
    try {
      const url = `${this.config.baseUrl}/api/mandates/${mandate_id}`;
      const response = await this.fetchWithTimeout(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get mandate config: ${response.status} ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      logger.error(`Error getting mandate config: ${error}`);
      throw error;
    }
  }

  /**
   * Fetch with timeout.
   */
  private async fetchWithTimeout(
    url: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.config.timeout}ms`);
      }
      throw error;
    }
  }
}

/**
 * Get CorporateSwarm client from environment configuration.
 */
export function getCorporateSwarmClient(): CorporateSwarmClient | null {
  const baseUrl = process.env.CORPORATE_SWARM_BASE_URL || process.env.VITE_CORPORATE_SWARM_BASE_URL;
  
  if (!baseUrl) {
    logger.warn('CorporateSwarm base URL not configured, client will not be available');
    return null;
  }

  return new CorporateSwarmClient({
    baseUrl,
    apiKey: process.env.CORPORATE_SWARM_API_KEY || process.env.VITE_CORPORATE_SWARM_API_KEY,
    timeout: parseInt(process.env.CORPORATE_SWARM_TIMEOUT || '30000', 10),
  });
}

