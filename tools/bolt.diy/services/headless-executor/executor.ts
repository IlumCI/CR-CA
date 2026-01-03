/**
 * Core execution logic for headless browser mandate execution.
 * 
 * Uses Playwright to control a browser and execute mandates via WebContainer.
 */

import { chromium, Browser, Page, BrowserContext } from 'playwright';
import type { Mandate, ExecutionEvent, ExecutionResult } from './types.js';

export interface ExecutorConfig {
  boltDiyUrl: string;
  governorUrl: string;
  workerId: string;
  headless: boolean;
  timeout: number;
}

export class HeadlessExecutor {
  private browser: Browser | null = null;
  private context: BrowserContext | null = null;
  private config: ExecutorConfig;
  private activePages: Map<string, Page> = new Map();

  constructor(config: ExecutorConfig) {
    this.config = config;
  }

  /**
   * Initialize browser and context.
   */
  async initialize(): Promise<void> {
    if (this.browser) {
      return; // Already initialized
    }

    this.browser = await chromium.launch({
      headless: this.config.headless !== false,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
      ],
    });

    this.context = await this.browser.newContext({
      viewport: { width: 1920, height: 1080 },
      ignoreHTTPSErrors: true,
    });

    console.log('Headless browser initialized');
  }

  /**
   * Execute a mandate in headless browser.
   */
  async executeMandate(mandate: Mandate): Promise<ExecutionResult> {
    if (!this.browser || !this.context) {
      await this.initialize();
    }

    if (!this.context) {
      throw new Error('Browser context not initialized');
    }

    const page = await this.context.newPage();
    this.activePages.set(mandate.mandate_id, page);

    try {
      // Navigate to execution page
      const executionUrl = `${this.config.boltDiyUrl}/execute/${mandate.mandate_id}`;
      console.log(`Navigating to ${executionUrl}`);
      
      await page.goto(executionUrl, {
        waitUntil: 'networkidle',
        timeout: 30000,
      });

      // Wait for page to load and start execution
      await page.waitForSelector('#root', { timeout: 10000 });

      // Inject mandate data into page
      await page.evaluate((mandateData) => {
        (window as any).__MANDATE_DATA__ = mandateData;
      }, mandate);

      // Wait for execution to complete
      // The page will auto-execute and emit events
      const result = await this.waitForExecutionComplete(page, mandate.mandate_id);

      return result;
    } catch (error) {
      console.error(`Error executing mandate ${mandate.mandate_id}:`, error);
      throw error;
    } finally {
      await page.close();
      this.activePages.delete(mandate.mandate_id);
    }
  }

  /**
   * Wait for execution to complete and collect result.
   */
  private async waitForExecutionComplete(
    page: Page,
    mandateId: string
  ): Promise<ExecutionResult> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Execution timeout for mandate ${mandateId}`));
      }, this.config.timeout);

      // Listen for execution completion event
      page.on('console', async (msg) => {
        const text = msg.text();
        if (text.includes('EXECUTION_COMPLETE')) {
          clearTimeout(timeout);
          try {
            // Get result from page
            const result = await page.evaluate(() => {
              return (window as any).__EXECUTION_RESULT__;
            });

            if (result) {
              resolve(result);
            } else {
              reject(new Error('Execution completed but no result found'));
            }
          } catch (error) {
            reject(error);
          }
        } else if (text.includes('EXECUTION_ERROR')) {
          clearTimeout(timeout);
          const error = await page.evaluate(() => {
            return (window as any).__EXECUTION_ERROR__;
          });
          reject(new Error(error || 'Execution failed'));
        }
      });

      // Also check for completion via DOM
      const checkInterval = setInterval(async () => {
        try {
          const status = await page.evaluate(() => {
            const statusEl = document.querySelector('[data-execution-status]');
            return statusEl?.getAttribute('data-execution-status');
          });

          if (status === 'completed' || status === 'failed') {
            clearInterval(checkInterval);
            clearTimeout(timeout);

            const result = await page.evaluate(() => {
              return (window as any).__EXECUTION_RESULT__;
            });

            if (result) {
              resolve(result);
            } else {
              // Create result from status
              resolve({
                mandate_id: mandateId,
                status: status === 'completed' ? 'success' : 'failed',
                iterations_completed: 0,
                final_state: {},
                governance_summary: {},
                budget_summary: {
                  tokens_used: 0,
                  time_elapsed: 0,
                  cost_incurred: 0,
                  budget_remaining: { tokens: 0, time: 0, cost: 0 },
                },
                events: [],
                created_at: Date.now(),
                completed_at: Date.now(),
              });
            }
          }
        } catch (error) {
          // Continue checking
        }
      }, 1000);
    });
  }

  /**
   * Cleanup browser and context.
   */
  async cleanup(): Promise<void> {
    // Close all active pages
    for (const page of this.activePages.values()) {
      await page.close().catch(() => {});
    }
    this.activePages.clear();

    if (this.context) {
      await this.context.close();
      this.context = null;
    }

    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }

    console.log('Headless browser cleaned up');
  }
}

