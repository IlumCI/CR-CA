/**
 * Priority Queue for Mandates
 * 
 * Manages mandate queue with priority-based ordering.
 */

import { Heap } from 'heap-js';
import type { QueuedMandate } from './types';

/**
 * Priority queue for mandates.
 * Higher priority values are dequeued first.
 */
export class MandatePriorityQueue {
  private heap: Heap<QueuedMandate>;
  private mandateMap: Map<string, QueuedMandate>;

  constructor() {
    // Max heap: higher priority = dequeued first
    this.heap = new Heap<QueuedMandate>((a, b) => {
      // First compare by priority (higher first)
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      // Then by queued time (earlier first)
      return a.queuedAt - b.queuedAt;
    });
    this.mandateMap = new Map();
  }

  /**
   * Enqueue a mandate with priority.
   */
  enqueue(mandate: QueuedMandate): void {
    this.heap.push(mandate);
    this.mandateMap.set(mandate.mandate.mandate_id, mandate);
  }

  /**
   * Dequeue the highest priority mandate.
   */
  dequeue(): QueuedMandate | undefined {
    const mandate = this.heap.pop();
    if (mandate) {
      this.mandateMap.delete(mandate.mandate.mandate_id);
    }
    return mandate;
  }

  /**
   * Peek at the highest priority mandate without removing it.
   */
  peek(): QueuedMandate | undefined {
    return this.heap.peek();
  }

  /**
   * Get mandate by ID.
   */
  get(mandateId: string): QueuedMandate | undefined {
    return this.mandateMap.get(mandateId);
  }

  /**
   * Remove mandate by ID.
   */
  remove(mandateId: string): boolean {
    const mandate = this.mandateMap.get(mandateId);
    if (!mandate) {
      return false;
    }
    
    // Rebuild heap without this mandate
    const mandates = Array.from(this.mandateMap.values()).filter(
      (m) => m.mandate.mandate_id !== mandateId
    );
    this.heap.clear();
    this.mandateMap.clear();
    
    for (const m of mandates) {
      this.heap.push(m);
      this.mandateMap.set(m.mandate.mandate_id, m);
    }
    
    return true;
  }

  /**
   * Get queue size.
   */
  size(): number {
    return this.heap.size();
  }

  /**
   * Check if queue is empty.
   */
  isEmpty(): boolean {
    return this.heap.size() === 0;
  }

  /**
   * Get queue position for a mandate (0 = next to execute).
   */
  getPosition(mandateId: string): number {
    const mandate = this.mandateMap.get(mandateId);
    if (!mandate) {
      return -1;
    }

    // Count mandates with higher priority or same priority but earlier
    let position = 0;
    for (const m of this.heap.iterator()) {
      if (m.mandate.mandate_id === mandateId) {
        break;
      }
      if (
        m.priority > mandate.priority ||
        (m.priority === mandate.priority && m.queuedAt < mandate.queuedAt)
      ) {
        position++;
      }
    }
    return position;
  }

  /**
   * Get all mandates in priority order (for debugging/monitoring).
   */
  getAll(): QueuedMandate[] {
    return Array.from(this.mandateMap.values()).sort((a, b) => {
      if (b.priority !== a.priority) {
        return b.priority - a.priority;
      }
      return a.queuedAt - b.queuedAt;
    });
  }
}

