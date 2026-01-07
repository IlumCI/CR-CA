/**
 * Mandate Executor for autonomous code generation cycles.
 * 
 * Orchestrates the complete execution lifecycle of a mandate:
 * - Plan generation via LLM
 * - Action execution via ActionRunner
 * - Iteration loops with quality checks
 * - Governance compliance checks
 * - Deployment (if configured)
 */

import type { WebContainer } from '@webcontainer/api';
import type { Mandate, ExecutionResult, ExecutionEvent, BuildOutput, FileDiff } from '~/types/mandate';
import { ExecutionEventEmitter, eventRegistry } from './execution-events';
import { ActionRunner } from './action-runner';
import { StreamingMessageParser, type ActionCallbackData } from './message-parser';
import { WORK_DIR } from '~/utils/constants';
import { createScopedLogger } from '~/utils/logger';
import type { IProviderSetting } from '~/types/model';
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from '~/utils/constants';
import { getSystemPrompt } from '~/lib/common/prompts/prompts';
import { BoltShell } from '~/utils/shell';
import { performCausalAnalysis, formatCausalAnalysisEvent } from '~/lib/governance/causal-analysis';
import { CorporateSwarmClient } from '~/lib/governance/corporate-swarm-client';

const logger = createScopedLogger('mandate-executor');

/**
 * Mandate executor for autonomous code generation.
 */
export class MandateExecutor {
  private mandate: Mandate;
  private webcontainer: Promise<WebContainer>;
  private shellTerminal: () => BoltShell;
  private eventEmitter: ExecutionEventEmitter;
  private actionRunner: ActionRunner;
  private budgetConsumed: {
    tokens: number;
    time: number;
    cost: number;
  };
  private startTime: number;
  private filesCreated: Set<string> = new Set();
  private filesModified: Set<string> = new Set();
  private apiKeys: Record<string, string>;
  private providerSettings: Record<string, IProviderSetting>;
  private env?: Env;
  private corporateSwarmClient: CorporateSwarmClient;

  constructor(
    mandate: Mandate,
    webcontainer: Promise<WebContainer>,
    shellTerminal: () => BoltShell,
    apiKeys: Record<string, string>,
    providerSettings: Record<string, IProviderSetting>,
    env?: Env
  ) {
    const constructorStartTime = Date.now();
    
    this.mandate = mandate;
    this.webcontainer = webcontainer;
    this.shellTerminal = shellTerminal;
    this.apiKeys = apiKeys;
    this.providerSettings = providerSettings;
    this.env = env;
    
    // Initialize event emitter
    this.eventEmitter = eventRegistry.getEmitter(mandate.mandate_id);
    this.eventEmitter.emitLog('debug', 'Initializing MandateExecutor...', 'mandate-executor');
    
    // Validate mandate
    this.eventEmitter.emitLog('debug', 'Validating mandate structure...', 'mandate-executor');
    if (!mandate.mandate_id || !mandate.objectives || !mandate.constraints || !mandate.budget) {
      this.eventEmitter.emitError('Invalid mandate structure: missing required fields', 'mandate-executor');
      throw new Error('Invalid mandate structure');
    }
    this.eventEmitter.emitLog('info', `Mandate validated: ${mandate.objectives.length} objective(s)`, 'mandate-executor');
    
    // Initialize CorporateSwarm client for governance integration
    const corporateSwarmUrl = process.env.CORPORATE_SWARM_API_URL || 'http://localhost:8000';
    this.eventEmitter.emitLog('debug', `Initializing CorporateSwarm client at ${corporateSwarmUrl}`, 'mandate-executor');
    this.corporateSwarmClient = new CorporateSwarmClient({ baseUrl: corporateSwarmUrl });
    this.eventEmitter.emitLog('info', 'CorporateSwarm client initialized', 'mandate-executor');

    this.budgetConsumed = {
      tokens: 0,
      time: 0,
      cost: 0,
    };
    this.startTime = Date.now();
    this.eventEmitter.emitLog('debug', 'Budget tracking initialized', 'mandate-executor');

    // Initialize action runner
    this.eventEmitter.emitLog('debug', 'Initializing ActionRunner...', 'mandate-executor');
    this.actionRunner = new ActionRunner(
      webcontainer,
      shellTerminal,
      (alert) => {
        this.eventEmitter.emitLog('warn', `Alert: ${alert.title} - ${alert.description}`, 'action-runner');
      },
      undefined, // supabase alert
      (alert) => {
        this.eventEmitter.emitLog('info', `Deployment: ${alert.title}`, 'deployment');
      }
    );
    this.eventEmitter.emitLog('info', 'ActionRunner initialized', 'mandate-executor');
    
    const constructorTime = Date.now() - constructorStartTime;
    this.eventEmitter.emitLog('info', `MandateExecutor constructor completed in ${constructorTime}ms`, 'mandate-executor');
  }

  /**
   * Execute the mandate through autonomous iteration cycles.
   */
  async execute(): Promise<ExecutionResult> {
    const executionStartTime = Date.now();
    logger.info(`Starting execution of mandate ${this.mandate.mandate_id}`);
    this.eventEmitter.emitLog('info', `Starting execution of mandate ${this.mandate.mandate_id}`, 'mandate-executor');
    this.eventEmitter.emitLog('info', `Budget limits: ${this.mandate.budget.token} tokens, ${this.mandate.budget.time}s time, $${this.mandate.budget.cost} cost`, 'mandate-executor');
    this.eventEmitter.emitLog('info', `Max iterations: ${this.mandate.iteration_config.max_iterations}, Quality threshold: ${this.mandate.iteration_config.quality_threshold}`, 'mandate-executor');

    const errors: ExecutionResult['errors'] = [];
    let finalStatus: ExecutionResult['status'] = 'success';
    let buildSuccessful = false;
    let testsPassed: boolean | undefined = undefined;
    let qualityScore: number | undefined = undefined;

    try {
      // Execute iterations
      this.eventEmitter.emitLog('info', `Starting iteration loop (max ${this.mandate.iteration_config.max_iterations} iterations)`, 'mandate-executor');
      for (
        let iteration = 0;
        iteration < this.mandate.iteration_config.max_iterations;
        iteration++
      ) {
        this.eventEmitter.setIteration(iteration);
        this.eventEmitter.emitIterationStart(iteration);
        this.eventEmitter.emitLog('info', `=== Iteration ${iteration + 1}/${this.mandate.iteration_config.max_iterations} ===`, 'mandate-executor');

        const iterationStartTime = Date.now();

        try {
          // Check budget before iteration
          if (!this.checkBudget()) {
            logger.warn(`Budget exhausted at iteration ${iteration}`);
            this.eventEmitter.emitBudgetWarning(this.budgetConsumed);
            finalStatus = 'budget_exceeded';
            break;
          }

          // 1. Plan phase - Generate plan using LLM
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Phase 1: Generating plan...`, 'mandate-executor');
          const planStartTime = Date.now();
          const plan = await this.generatePlan(iteration);
          if (!plan) {
            throw new Error('Failed to generate plan');
          }
          const planTime = Date.now() - planStartTime;
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Plan generated in ${planTime}ms`, 'mandate-executor');

          // 2. Apply phase - Execute actions from plan
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Phase 2: Executing plan...`, 'mandate-executor');
          const applyStartTime = Date.now();
          const actionsExecuted = await this.executePlan(plan, iteration);
          const applyTime = Date.now() - applyStartTime;
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Plan executed in ${applyTime}ms (${actionsExecuted.length} actions)`, 'mandate-executor');

          // 3. Test phase (if required)
          if (this.mandate.iteration_config.test_required) {
            this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Phase 3: Running tests...`, 'mandate-executor');
            const testStartTime = Date.now();
            testsPassed = await this.runTests();
            const testTime = Date.now() - testStartTime;
            this.eventEmitter.emitLog(
              testsPassed ? 'info' : 'warn',
              `[Iteration ${iteration}] Tests ${testsPassed ? 'passed' : 'failed'} in ${testTime}ms`,
              'testing'
            );
          } else {
            this.eventEmitter.emitLog('debug', `[Iteration ${iteration}] Phase 3: Tests skipped (not required)`, 'mandate-executor');
          }

          // 4. Evaluate phase - Quality check
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Phase 4: Evaluating quality...`, 'mandate-executor');
          const evalStartTime = Date.now();
          qualityScore = await this.evaluateQuality(iteration);
          const evalTime = Date.now() - evalStartTime;
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Quality score: ${qualityScore} (evaluated in ${evalTime}ms)`, 'evaluation');

          // 5. Governance check
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Phase 5: Performing governance check...`, 'mandate-executor');
          const govStartTime = Date.now();
          await this.performGovernanceCheck(iteration);
          const govTime = Date.now() - govStartTime;
          this.eventEmitter.emitLog('info', `[Iteration ${iteration}] Governance check completed in ${govTime}ms`, 'mandate-executor');

          // 6. Check if we should continue
          if (qualityScore >= this.mandate.iteration_config.quality_threshold) {
            logger.info(`Quality threshold met at iteration ${iteration}`);
            this.eventEmitter.emitIterationEnd(iteration, 'success', {
              execution_time: Date.now() - iterationStartTime,
            });
            break;
          }

          // Check if we should stop on error
          if (
            this.mandate.iteration_config.stop_on_error &&
            actionsExecuted.some((a) => a.status === 'failed')
          ) {
            logger.warn(`Stopping due to error at iteration ${iteration}`);
            finalStatus = 'failed';
            this.eventEmitter.emitIterationEnd(iteration, 'failed', {
              execution_time: Date.now() - iterationStartTime,
            });
            break;
          }

          this.eventEmitter.emitIterationEnd(iteration, 'success', {
            execution_time: Date.now() - iterationStartTime,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`Iteration ${iteration} failed: ${errorMessage}`);

          errors.push({
            iteration,
            type: 'execution_error',
            message: errorMessage,
            timestamp: Date.now(),
          });

          this.eventEmitter.emitError(errorMessage, 'mandate-executor', {
            execution_time: Date.now() - iterationStartTime,
          });

          if (this.mandate.iteration_config.stop_on_error) {
            finalStatus = 'failed';
            this.eventEmitter.emitIterationEnd(iteration, 'failed', {
              execution_time: Date.now() - iterationStartTime,
            });
            break;
          }

          // Retry logic
          if (
            this.mandate.iteration_config.retry_on_failure &&
            iteration < (this.mandate.iteration_config.max_retries || 3)
          ) {
            logger.info(`Retrying iteration ${iteration}`);
            continue;
          }

          this.eventEmitter.emitIterationEnd(iteration, 'failed', {
            execution_time: Date.now() - iterationStartTime,
          });
        }

        // Update time budget
        this.budgetConsumed.time = (Date.now() - this.startTime) / 1000;
      }

      // Build phase (if needed for deployment)
      if (this.mandate.deployment?.enabled) {
        try {
          const buildOutput = await this.buildProject();
          buildSuccessful = buildOutput.success;
          this.eventEmitter.emitLog(
            buildSuccessful ? 'info' : 'error',
            `Build ${buildSuccessful ? 'succeeded' : 'failed'}`,
            'build'
          );
        } catch (error) {
          logger.error('Build failed:', error);
          buildSuccessful = false;
        }
      }

      // Deployment phase (if enabled)
      let deploymentResult: ExecutionResult['deployment_result'] | undefined = undefined;
      if (this.mandate.deployment?.enabled && this.mandate.deployment.auto_deploy && buildSuccessful) {
        try {
          deploymentResult = await this.deployProject();
        } catch (error) {
          logger.error('Deployment failed:', error);
          deploymentResult = {
            deployed: false,
            provider: this.mandate.deployment.provider,
            error: error instanceof Error ? error.message : String(error),
          };
        }
      }

      // Final governance check
      const governanceSummary = await this.getGovernanceSummary();

      const result: ExecutionResult = {
        mandate_id: this.mandate.mandate_id,
        status: finalStatus,
        iterations_completed: this.eventEmitter.getEvents().filter((e) => e.type === 'iteration_end').length,
        final_state: {
          files_created: Array.from(this.filesCreated),
          files_modified: Array.from(this.filesModified),
          build_successful: buildSuccessful,
          tests_passed: testsPassed,
          quality_score: qualityScore,
        },
        governance_summary: governanceSummary,
        budget_summary: {
          tokens_used: this.budgetConsumed.tokens,
          time_elapsed: this.budgetConsumed.time,
          cost_incurred: this.budgetConsumed.cost,
          budget_remaining: {
            tokens: Math.max(0, this.mandate.budget.token - this.budgetConsumed.tokens),
            time: Math.max(0, this.mandate.budget.time - this.budgetConsumed.time),
            cost: Math.max(0, this.mandate.budget.cost - this.budgetConsumed.cost),
          },
        },
        deployment_result: deploymentResult,
        errors: errors.length > 0 ? errors : undefined,
        events: this.eventEmitter.getEvents(),
        created_at: this.startTime,
        completed_at: Date.now(),
      };

      logger.info(`Mandate ${this.mandate.mandate_id} execution completed with status: ${finalStatus}`);

      return result;
    } catch (error) {
      logger.error(`Fatal error executing mandate ${this.mandate.mandate_id}:`, error);
      finalStatus = 'failed';

      return {
        mandate_id: this.mandate.mandate_id,
        status: finalStatus,
        iterations_completed: this.eventEmitter.getEvents().filter((e) => e.type === 'iteration_end').length,
        final_state: {
          files_created: Array.from(this.filesCreated),
          files_modified: Array.from(this.filesModified),
          build_successful: false,
        },
        governance_summary: {},
        budget_summary: {
          tokens_used: this.budgetConsumed.tokens,
          time_elapsed: this.budgetConsumed.time,
          cost_incurred: this.budgetConsumed.cost,
          budget_remaining: {
            tokens: Math.max(0, this.mandate.budget.token - this.budgetConsumed.tokens),
            time: Math.max(0, this.mandate.budget.time - this.budgetConsumed.time),
            cost: Math.max(0, this.mandate.budget.cost - this.budgetConsumed.cost),
          },
        },
        errors: [
          {
            iteration: -1,
            type: 'fatal_error',
            message: error instanceof Error ? error.message : String(error),
            timestamp: Date.now(),
          },
        ],
        events: this.eventEmitter.getEvents(),
        created_at: this.startTime,
        completed_at: Date.now(),
      };
    }
  }

  /**
   * Generate execution plan using LLM.
   */
  private async generatePlan(iteration: number): Promise<string | null> {
    logger.debug(`Generating plan for iteration ${iteration}`);

    const objectivesText = this.mandate.objectives.join('\n- ');
    const deliverablesText = this.mandate.deliverables.join('\n- ');
    const constraintsText = this.formatConstraints();

    const mandatePrompt = `
You are executing a corporate mandate for autonomous code generation. This is iteration ${iteration + 1} of ${this.mandate.iteration_config.max_iterations}.

MANDATE OBJECTIVES:
${objectivesText}

DELIVERABLES REQUIRED:
${deliverablesText}

CONSTRAINTS:
${constraintsText}

${this.mandate.governance?.causal_analysis_required ? '\nCAUSAL ANALYSIS REQUIRED: You must consider causal relationships and counterfactual scenarios in your implementation.\n' : ''}

Generate a comprehensive plan and implementation using <boltArtifact> tags. Follow all coding best practices and ensure the code is production-ready.

Current working directory: ${WORK_DIR}
`;

    try {
      const messages: Messages = [
        {
          id: `mandate-${this.mandate.mandate_id}-plan-${Date.now()}`,
          role: 'user',
          content: mandatePrompt,
        },
      ];

      const systemPrompt = getSystemPrompt(WORK_DIR, undefined, undefined);

      // Dynamically import streamText using eval to avoid static analysis
      // This prevents Remix from detecting the server-only import during build
      // Note: This will only work on the server; client-side code should use API routes
      const streamTextModule = await eval('import("~/lib/.server/llm/stream-text")');
      const { streamText } = streamTextModule;
      
      const result = await streamText({
        messages,
        env: this.env,
        apiKeys: this.apiKeys,
        providerSettings: this.providerSettings,
        options: {
          system: systemPrompt,
          maxSteps: 10,
        },
        chatMode: 'build',
      });

      // Collect the full response
      let fullResponse = '';
      for await (const chunk of result.textStream) {
        fullResponse += chunk;
      }

      // Track token usage
      // Note: Actual usage tracking would need to be extracted from result
      // For now, estimate based on response length
      const estimatedTokens = Math.ceil(fullResponse.length / 4); // Rough estimate
      this.budgetConsumed.tokens += estimatedTokens;

      this.eventEmitter.emitLog('info', `Plan generated (estimated ${estimatedTokens} tokens)`, 'llm');

      return fullResponse;
    } catch (error) {
      logger.error('Failed to generate plan:', error);
      this.eventEmitter.emitError('Failed to generate plan', 'llm');
      return null;
    }
  }

  /**
   * Format constraints for prompt.
   */
  private formatConstraints(): string {
    const constraints = this.mandate.constraints;
    const parts: string[] = [];

    parts.push(`- Language: ${constraints.language}`);
    parts.push(`- Max dependencies: ${constraints.maxDependencies}`);
    parts.push(`- Network access: ${constraints.noNetwork ? 'DISABLED' : 'ENABLED'}`);
    parts.push(`- Max file size: ${constraints.maxFileSize} bytes`);
    parts.push(`- Max files: ${constraints.maxFiles}`);

    if (constraints.allowedPackages && constraints.allowedPackages.length > 0) {
      parts.push(`- Allowed packages: ${constraints.allowedPackages.join(', ')}`);
    }

    if (constraints.forbiddenPatterns && constraints.forbiddenPatterns.length > 0) {
      parts.push(`- Forbidden patterns: ${constraints.forbiddenPatterns.join(', ')}`);
    }

    return parts.join('\n');
  }

  /**
   * Execute plan by parsing artifacts and running actions.
   */
  private async executePlan(plan: string, iteration: number): Promise<Array<{ status: string }>> {
    logger.debug(`Executing plan for iteration ${iteration}`);

    const parser = new StreamingMessageParser({
      callbacks: {
        onActionClose: async (data: ActionCallbackData) => {
          // Validate constraints before execution
          if (!this.validateActionConstraints(data.action)) {
            this.eventEmitter.emitConstraintViolation(
              'constraint_violation',
              `Action violates mandate constraints: ${data.action.type}`
            );
            return;
          }

          // Execute action via ActionRunner
          await this.actionRunner.runAction(data, data.action.type === 'file');
        },
      },
    });

    // Parse the plan to extract actions
    const messageId = `mandate-${this.mandate.mandate_id}-iter-${iteration}`;
    parser.parse(messageId, plan);

    // Track file changes
    const actions = this.actionRunner.actions.get();
    const executedActions: Array<{ status: string }> = [];

    for (const [actionId, action] of Object.entries(actions)) {
      if (action.type === 'file') {
        if (!this.filesCreated.has(action.filePath) && !this.filesModified.has(action.filePath)) {
          // Check if file exists
          try {
            const webcontainer = await this.webcontainer;
            await webcontainer.fs.readFile(action.filePath);
            this.filesModified.add(action.filePath);
          } catch {
            this.filesCreated.add(action.filePath);
          }
        } else {
          this.filesModified.add(action.filePath);
        }
      }

      executedActions.push({
        status: action.status,
      });
    }

    // Emit diff event
    const filesChanged = Array.from(new Set([...this.filesCreated, ...this.filesModified]));
    if (filesChanged.length > 0) {
      this.eventEmitter.emitDiff(filesChanged);
    }

    return executedActions;
  }

  /**
   * Validate action against mandate constraints.
   */
  private validateActionConstraints(action: any): boolean {
    // Check max files constraint
    const totalFiles = this.filesCreated.size + this.filesModified.size;
    if (totalFiles >= this.mandate.constraints.maxFiles) {
      return false;
    }

    // Check file size constraint for file actions
    if (action.type === 'file' && action.content) {
      const fileSize = new Blob([action.content]).size;
      if (fileSize > this.mandate.constraints.maxFileSize) {
        return false;
      }
    }

    // Check network constraint for shell actions
    if (action.type === 'shell' && this.mandate.constraints.noNetwork) {
      const networkCommands = ['curl', 'wget', 'fetch', 'http', 'https'];
      if (networkCommands.some((cmd) => action.content.includes(cmd))) {
        return false;
      }
    }

    // Check allowed packages constraint
    if (action.type === 'shell' && action.content.includes('npm install')) {
      if (this.mandate.constraints.allowedPackages && this.mandate.constraints.allowedPackages.length > 0) {
        // Extract package names from npm install command
        const packageMatch = action.content.match(/npm install[^&]*?([^\s&]+)/g);
        if (packageMatch) {
          for (const match of packageMatch) {
            const packageName = match.replace('npm install', '').trim();
            if (
              !this.mandate.constraints.allowedPackages.some((allowed) =>
                packageName.includes(allowed)
              )
            ) {
              return false;
            }
          }
        }
      }

      // Check max dependencies
      const depCount = (action.content.match(/npm install/g) || []).length;
      if (depCount > this.mandate.constraints.maxDependencies) {
        return false;
      }
    }

    return true;
  }

  /**
   * Run tests if required.
   */
  private async runTests(): Promise<boolean> {
    try {
      const shell = this.shellTerminal();
      await shell.ready();

      // Try common test commands
      const testCommands = ['npm test', 'npm run test', 'yarn test', 'pnpm test'];
      for (const cmd of testCommands) {
        try {
          const result = await shell.executeCommand(`mandate-${this.mandate.mandate_id}`, cmd);
          if (result?.exitCode === 0) {
            return true;
          }
        } catch {
          continue;
        }
      }

      return false;
    } catch (error) {
      logger.warn('Test execution failed:', error);
      return false;
    }
  }

  /**
   * Evaluate quality of current iteration.
   */
  private async evaluateQuality(iteration: number): Promise<number> {
    // Simplified quality evaluation
    // In a full implementation, this would use LLM to evaluate code quality
    let score = 0.5; // Base score

    // Increase score if build succeeds
    try {
      const buildOutput = await this.buildProject();
      if (buildOutput.success) {
        score += 0.3;
      }
    } catch {
      // Build failed, don't increase score
    }

    // Increase score if tests pass
    if (this.mandate.iteration_config.test_required) {
      const testsPassed = await this.runTests();
      if (testsPassed) {
        score += 0.2;
      }
    }

    return Math.min(1.0, score);
  }

  /**
   * Perform governance check.
   */
  private async performGovernanceCheck(iteration: number): Promise<void> {
    logger.debug(`Performing governance check for iteration ${iteration}`);
    
    // Collect code changes for causal analysis
    const codeChanges: Array<{ file: string; content: string }> = [];
    for (const file of this.filesModified) {
      try {
        const webcontainer = await this.webcontainer;
        const fileContent = await webcontainer.fs.readFile(file, 'utf-8');
        codeChanges.push({ file, content: fileContent });
      } catch (error) {
        logger.warn(`Failed to read file ${file} for causal analysis: ${error}`);
      }
    }
    
    // Perform causal analysis if required
    let causalAnalysisResult = null;
    if (this.mandate.governance?.causal_analysis_required && codeChanges.length > 0) {
      logger.info(`Performing causal analysis for iteration ${iteration}`);
      causalAnalysisResult = await performCausalAnalysis(
        this.mandate,
        codeChanges,
        this.corporateSwarmClient
      );
      
      if (causalAnalysisResult) {
        // Emit causal analysis event
        const causalEvent = formatCausalAnalysisEvent(
          this.mandate.mandate_id,
          iteration,
          causalAnalysisResult
        );
        this.eventEmitter.emit(causalEvent.type, causalEvent.data, causalEvent.metadata);
      }
    }
    
    // Emit governance check event
    this.eventEmitter.emitGovernanceCheck(
      this.mandate.governance?.esg_requirements,
      undefined, // risk assessment - to be calculated
      {
        tokens: this.budgetConsumed.tokens,
        time: this.budgetConsumed.time,
        cost: this.budgetConsumed.cost,
      }
    );
  }

  /**
   * Get governance summary.
   */
  private async getGovernanceSummary(): Promise<ExecutionResult['governance_summary']> {
    // Placeholder - will be implemented with governance hooks
    return {
      esg_score: this.mandate.governance?.esg_requirements,
      compliance_status: {},
    };
  }

  /**
   * Check if budget is still available.
   */
  private checkBudget(): boolean {
    return (
      this.budgetConsumed.tokens < this.mandate.budget.token &&
      this.budgetConsumed.time < this.mandate.budget.time &&
      this.budgetConsumed.cost < this.mandate.budget.cost
    );
  }

  /**
   * Build the project.
   */
  private async buildProject(): Promise<BuildOutput> {
    try {
      const shell = this.shellTerminal();
      await shell.ready();

      // Try common build commands
      const buildCommands = ['npm run build', 'npm run build:prod', 'yarn build', 'pnpm build'];
      let buildOutput = '';
      let buildErrors: string[] = [];

      for (const cmd of buildCommands) {
        try {
          const result = await shell.executeCommand(`mandate-${this.mandate.mandate_id}`, cmd);
          buildOutput = result?.output || '';
          if (result?.exitCode === 0) {
            return {
              success: true,
              output: buildOutput,
              build_time: Date.now() - this.startTime,
            };
          } else {
            buildErrors.push(buildOutput);
          }
        } catch (error) {
          buildErrors.push(error instanceof Error ? error.message : String(error));
          continue;
        }
      }

      return {
        success: false,
        output: buildOutput,
        errors: buildErrors,
      };
    } catch (error) {
      return {
        success: false,
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }

  /**
   * Deploy project to configured provider.
   */
  private async deployProject(): Promise<ExecutionResult['deployment_result']> {
    if (!this.mandate.deployment?.enabled) {
      return { deployed: false };
    }

    const provider = this.mandate.deployment.provider;
    this.eventEmitter.emitDeploymentStart(provider);

    try {
      // Build first
      const buildOutput = await this.buildProject();
      if (!buildOutput.success) {
        throw new Error('Build failed before deployment');
      }

      // Call appropriate deployment API
      // Note: These APIs will be called via HTTP, not directly
      // For now, return a placeholder
      const deploymentUrl = await this.callDeploymentAPI(provider, buildOutput);

      this.eventEmitter.emitDeploymentEnd(provider, 'success', deploymentUrl);

      return {
        deployed: true,
        provider,
        url: deploymentUrl,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.eventEmitter.emitDeploymentEnd(provider, 'failed', undefined, errorMessage);

      return {
        deployed: false,
        provider,
        error: errorMessage,
      };
    }
  }

  /**
   * Call deployment API for the specified provider.
   */
  private async callDeploymentAPI(provider: string, buildOutput: BuildOutput): Promise<string> {
    logger.info(`Deploying to ${provider} for mandate ${this.mandate.mandate_id}`);

    const baseUrl = typeof window !== 'undefined' 
      ? window.location.origin 
      : process.env.VITE_PUBLIC_APP_URL || 'http://localhost:5173';
    const deployEndpoint = `/api/${provider}-deploy`;

    try {
      // Get all files from webcontainer
      const webcontainer = await this.webcontainer;
      const files: Record<string, string> = {};
      
      // Collect all files from the project
      const collectFiles = async (dir: string = WORK_DIR): Promise<void> => {
        const entries = await webcontainer.fs.readdir(dir, { withFileTypes: true });
        
        for (const entry of entries) {
          const fullPath = `${dir}/${entry.name}`.replace(/\/+/g, '/');
          
          if (entry.isFile()) {
            try {
              const content = await webcontainer.fs.readFile(fullPath, 'utf-8');
              // Remove WORK_DIR prefix for deployment
              const relativePath = fullPath.replace(WORK_DIR, '').replace(/^\//, '');
              files[relativePath] = content;
            } catch (error) {
              logger.warn(`Failed to read file ${fullPath}: ${error}`);
            }
          } else if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
            await collectFiles(fullPath);
          }
        }
      };

      await collectFiles();

      // Prepare deployment request
      const deployRequest: any = {
        files,
        mandate_id: this.mandate.mandate_id,
        proposal_id: this.mandate.governance?.proposal_id,
      };

      // Add provider-specific requirements
      if (provider === 'netlify' || provider === 'vercel') {
        // These APIs require token from environment or stored credentials
        // For now, we'll pass through - the API will handle authentication
      }

      // Make deployment request
      const response = await fetch(`${baseUrl}${deployEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(deployRequest),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: response.statusText })) as { error?: string };
        throw new Error(errorData.error || `Deployment failed: ${response.statusText}`);
      }

      const result = await response.json() as { deploy?: { url?: string; ssl_url?: string }; site?: { url?: string }; url?: string };
      
      // Extract deployment URL from response
      const deploymentUrl = result.deploy?.url || result.deploy?.ssl_url || result.site?.url || result.url;
      
      if (!deploymentUrl) {
        throw new Error('Deployment succeeded but no URL returned');
      }

      logger.info(`Deployment successful to ${deploymentUrl}`);
      return deploymentUrl;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Deployment to ${provider} failed: ${errorMessage}`);
      throw error;
    }
  }
}

