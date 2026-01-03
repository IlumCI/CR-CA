/**
 * Governance hooks for mandate execution.
 * 
 * Provides ESG compliance checking, risk assessment, budget validation,
 * and approval chain enforcement for autonomous code generation.
 */

import type {
  Mandate,
  ESGScore,
  RiskAssessment,
  ExecutionEventData,
  FileDiff,
} from '~/types/mandate';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('governance-hooks');

/**
 * Governance hooks for mandate execution oversight.
 */
export class GovernanceHooks {
  /**
   * Check ESG compliance of generated code.
   */
  async checkESGCompliance(mandate: Mandate, code: string): Promise<ESGScore> {
    logger.debug('Checking ESG compliance');

    // If mandate has ESG requirements, use them as baseline
    const baseline = mandate.governance.esg_requirements || {
      environmental_score: 0.5,
      social_score: 0.5,
      governance_score: 0.5,
      overall_score: 0.5,
    };

    // Simple heuristic-based ESG scoring
    // In a full implementation, this would use LLM or specialized models
    let environmentalScore = baseline.environmental_score;
    let socialScore = baseline.social_score;
    let governanceScore = baseline.governance_score;

    // Environmental checks
    const envKeywords = {
      positive: ['renewable', 'sustainable', 'green', 'carbon', 'emission', 'efficiency'],
      negative: ['waste', 'pollution', 'depletion', 'toxic'],
    };

    const envPositiveCount = envKeywords.positive.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;
    const envNegativeCount = envKeywords.negative.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;

    environmentalScore += envPositiveCount * 0.05;
    environmentalScore -= envNegativeCount * 0.1;
    environmentalScore = Math.max(0, Math.min(1, environmentalScore));

    // Social checks
    const socialKeywords = {
      positive: ['accessibility', 'inclusive', 'diversity', 'equity', 'privacy', 'security'],
      negative: ['discriminat', 'bias', 'exclusive', 'exploit'],
    };

    const socialPositiveCount = socialKeywords.positive.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;
    const socialNegativeCount = socialKeywords.negative.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;

    socialScore += socialPositiveCount * 0.05;
    socialScore -= socialNegativeCount * 0.1;
    socialScore = Math.max(0, Math.min(1, socialScore));

    // Governance checks
    const governanceKeywords = {
      positive: ['audit', 'compliance', 'transparency', 'accountability', 'ethics'],
      negative: ['bypass', 'override', 'ignore', 'skip'],
    };

    const govPositiveCount = governanceKeywords.positive.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;
    const govNegativeCount = governanceKeywords.negative.filter((kw) =>
      code.toLowerCase().includes(kw)
    ).length;

    governanceScore += govPositiveCount * 0.05;
    governanceScore -= govNegativeCount * 0.1;
    governanceScore = Math.max(0, Math.min(1, governanceScore));

    const overallScore = (environmentalScore + socialScore + governanceScore) / 3;

    const esgScore: ESGScore = {
      environmental_score: environmentalScore,
      social_score: socialScore,
      governance_score: governanceScore,
      overall_score: overallScore,
      last_updated: Date.now(),
    };

    logger.debug(`ESG score: ${overallScore.toFixed(2)}`);

    return esgScore;
  }

  /**
   * Assess risk of code changes.
   */
  async assessRisk(mandate: Mandate, changes: FileDiff[]): Promise<RiskAssessment> {
    logger.debug('Assessing risk');

    let riskLevel: RiskAssessment['risk_level'] = 'low';
    let probability = 0.2;
    let impact = 0.2;

    // Analyze file changes
    const totalChanges = changes.reduce((sum, diff) => sum + diff.additions + diff.deletions, 0);
    const filesChanged = changes.length;

    // High change volume increases risk
    if (totalChanges > 1000) {
      probability += 0.3;
      impact += 0.2;
    } else if (totalChanges > 500) {
      probability += 0.2;
      impact += 0.1;
    }

    // Many files changed increases risk
    if (filesChanged > 20) {
      probability += 0.2;
      impact += 0.15;
    } else if (filesChanged > 10) {
      probability += 0.1;
      impact += 0.1;
    }

    // Check for risky file patterns
    const riskyPatterns = [
      /\.(env|config|secret|key|credential)/i,
      /database|db|sql/i,
      /auth|security|password/i,
      /payment|transaction|billing/i,
    ];

    const riskyFiles = changes.filter((diff) =>
      riskyPatterns.some((pattern) => pattern.test(diff.filepath))
    );

    if (riskyFiles.length > 0) {
      probability += 0.3;
      impact += 0.4;
    }

    // Calculate risk score
    probability = Math.min(1, probability);
    impact = Math.min(1, impact);
    const riskScore = probability * impact;

    // Determine risk level
    if (riskScore >= 0.7) {
      riskLevel = 'critical';
    } else if (riskScore >= 0.5) {
      riskLevel = 'high';
    } else if (riskScore >= 0.3) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'low';
    }

    // Check against mandate risk threshold
    if (mandate.governance.risk_threshold && riskScore > mandate.governance.risk_threshold) {
      riskLevel = 'high';
      probability = Math.max(probability, 0.6);
      impact = Math.max(impact, 0.6);
    }

    const assessment: RiskAssessment = {
      risk_category: 'code_changes',
      risk_level: riskLevel,
      probability,
      impact,
      risk_score: riskScore,
      mitigation_strategies: this.generateMitigationStrategies(riskLevel, changes),
      status: 'active',
      last_reviewed: Date.now(),
    };

    logger.debug(`Risk assessment: ${riskLevel} (score: ${riskScore.toFixed(2)})`);

    return assessment;
  }

  /**
   * Generate mitigation strategies based on risk level.
   */
  private generateMitigationStrategies(
    riskLevel: RiskAssessment['risk_level'],
    changes: FileDiff[]
  ): string[] {
    const strategies: string[] = [];

    if (riskLevel === 'critical' || riskLevel === 'high') {
      strategies.push('Conduct thorough code review before deployment');
      strategies.push('Run comprehensive test suite');
      strategies.push('Perform security audit');
      strategies.push('Implement gradual rollout');
    }

    if (riskLevel === 'critical') {
      strategies.push('Require executive approval');
      strategies.push('Enable enhanced monitoring');
      strategies.push('Prepare rollback plan');
    }

    // Check for specific risky patterns
    const hasAuthChanges = changes.some((diff) => /auth|login|password/i.test(diff.filepath));
    if (hasAuthChanges) {
      strategies.push('Review authentication and authorization logic');
    }

    const hasDataChanges = changes.some((diff) => /database|db|model|schema/i.test(diff.filepath));
    if (hasDataChanges) {
      strategies.push('Backup database before changes');
      strategies.push('Test data migration scripts');
    }

    return strategies;
  }

  /**
   * Check budget constraints.
   */
  async checkBudget(
    mandate: Mandate,
    currentCost: { tokens: number; time: number; cost: number }
  ): Promise<boolean> {
    const budget = mandate.budget;

    const withinBudget =
      currentCost.tokens <= budget.token &&
      currentCost.time <= budget.time &&
      currentCost.cost <= budget.cost;

    if (!withinBudget) {
      logger.warn(
        `Budget exceeded: tokens=${currentCost.tokens}/${budget.token}, time=${currentCost.time}/${budget.time}, cost=${currentCost.cost}/${budget.cost}`
      );
    }

    return withinBudget;
  }

  /**
   * Check if approval is required at a checkpoint.
   */
  async requireApproval(mandate: Mandate, checkpoint: string): Promise<boolean> {
    const approvalChain = mandate.governance.approval_chain;

    if (!approvalChain || approvalChain.length === 0) {
      return false; // No approval chain, no approval required
    }

    // Check if this checkpoint requires approval
    // In a full implementation, this would check against CorporateSwarm
    // For now, return true if approval chain exists
    logger.debug(`Approval required at checkpoint: ${checkpoint}`);

    return true;
  }

  /**
   * Check compliance with specified frameworks.
   */
  async checkCompliance(
    mandate: Mandate,
    code: string,
    frameworks?: string[]
  ): Promise<Record<string, boolean>> {
    const complianceFrameworks = frameworks || mandate.governance.compliance_frameworks || [];
    const compliance: Record<string, boolean> = {};

    for (const framework of complianceFrameworks) {
      compliance[framework] = this.checkFrameworkCompliance(framework, code);
    }

    return compliance;
  }

  /**
   * Check compliance with a specific framework.
   */
  private checkFrameworkCompliance(framework: string, code: string): boolean {
    const frameworkLower = framework.toLowerCase();

    // Simple heuristic-based compliance checking
    // In a full implementation, this would use specialized compliance checkers

    if (frameworkLower.includes('gdpr')) {
      // GDPR compliance checks
      const gdprKeywords = ['consent', 'privacy', 'data protection', 'right to be forgotten'];
      return gdprKeywords.some((keyword) => code.toLowerCase().includes(keyword));
    }

    if (frameworkLower.includes('sox')) {
      // SOX compliance checks
      const soxKeywords = ['audit', 'control', 'financial', 'reporting'];
      return soxKeywords.some((keyword) => code.toLowerCase().includes(keyword));
    }

    if (frameworkLower.includes('iso') && frameworkLower.includes('27001')) {
      // ISO 27001 compliance checks
      const isoKeywords = ['security', 'encryption', 'access control', 'incident'];
      return isoKeywords.some((keyword) => code.toLowerCase().includes(keyword));
    }

    // Default: assume compliant if framework is specified
    return true;
  }

  /**
   * Get governance summary for execution result.
   */
  async getGovernanceSummary(
    mandate: Mandate,
    code: string,
    changes: FileDiff[],
    budgetConsumed: { tokens: number; time: number; cost: number }
  ): Promise<{
    esg_score?: ESGScore;
    risk_assessment?: RiskAssessment;
    compliance_status?: Record<string, boolean>;
    budget_status: 'within_budget' | 'exceeded' | 'warning';
  }> {
    const esgScore = await this.checkESGCompliance(mandate, code);
    const riskAssessment = await this.assessRisk(mandate, changes);
    const complianceStatus = await this.checkCompliance(mandate, code);
    const budgetWithinLimit = await this.checkBudget(mandate, budgetConsumed);

    // Determine budget status
    const budgetUsage = {
      tokens: budgetConsumed.tokens / mandate.budget.token,
      time: budgetConsumed.time / mandate.budget.time,
      cost: budgetConsumed.cost / mandate.budget.cost,
    };

    const maxUsage = Math.max(budgetUsage.tokens, budgetUsage.time, budgetUsage.cost);
    let budgetStatus: 'within_budget' | 'exceeded' | 'warning';
    if (!budgetWithinLimit) {
      budgetStatus = 'exceeded';
    } else if (maxUsage > 0.8) {
      budgetStatus = 'warning';
    } else {
      budgetStatus = 'within_budget';
    }

    return {
      esg_score: esgScore,
      risk_assessment: riskAssessment,
      compliance_status: complianceStatus,
      budget_status: budgetStatus,
    };
  }
}

