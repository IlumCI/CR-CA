"""
Batch processing utilities.

Provides functionality for:
- Parallel task execution
- Batch rate limiting
- Progress tracking
- Error aggregation
"""

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from loguru import logger

try:
    from utils.rate_limiter import RateLimiter, RateLimitConfig
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    logger.debug("Rate limiter not available for batch processing")


@dataclass
class BatchResult:
    """Result of a batch processing operation.
    
    Attributes:
        task_id: Task identifier
        success: Whether task succeeded
        result: Task result (if successful)
        error: Error message (if failed)
        execution_time: Time taken to execute task
    """
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class BatchStats:
    """Statistics for batch processing.
    
    Attributes:
        total_tasks: Total number of tasks
        completed_tasks: Number of completed tasks
        failed_tasks: Number of failed tasks
        total_time: Total execution time
        average_time: Average execution time per task
    """
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time: float = 0.0
    average_time: float = 0.0


class BatchProcessor:
    """Batch processor for parallel task execution.
    
    Provides functionality for:
    - Parallel task execution with configurable workers
    - Batch rate limiting
    - Progress tracking
    - Error aggregation
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        rate_limiter: Optional[Any] = None,
        rate_limit_config: Optional[Any] = None,
    ):
        """Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            rate_limiter: Optional rate limiter instance
            rate_limit_config: Optional rate limit configuration
        """
        self.max_workers = max_workers
        
        # Set up rate limiting
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        elif RATE_LIMITER_AVAILABLE and rate_limit_config is not None:
            self.rate_limiter = RateLimiter(rate_limit_config)
        elif RATE_LIMITER_AVAILABLE:
            # Default rate limiting for batch processing
            config = RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=5000,
            )
            self.rate_limiter = RateLimiter(config)
        else:
            self.rate_limiter = None
        
        logger.debug(f"Initialized BatchProcessor with {max_workers} workers")
    
    def process_batch(
        self,
        tasks: List[Any],
        task_fn: Callable,
        task_ids: Optional[List[str]] = None,
        user_id: str = "default",
        show_progress: bool = True,
    ) -> Tuple[List[BatchResult], BatchStats]:
        """Process a batch of tasks in parallel.
        
        Args:
            tasks: List of task inputs
            task_fn: Function to execute for each task
            task_ids: Optional list of task identifiers
            user_id: User identifier for rate limiting
            show_progress: Whether to show progress updates
            
        Returns:
            Tuple of (results, stats)
        """
        if not tasks:
            return [], BatchStats()
        
        start_time = time.time()
        results: List[BatchResult] = []
        
        # Generate task IDs if not provided
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(tasks))]
        
        if len(task_ids) != len(tasks):
            logger.warning("Task IDs length doesn't match tasks length, generating new IDs")
            task_ids = [f"task_{i}" for i in range(len(tasks))]
        
        def process_single_task(task: Any, task_id: str) -> BatchResult:
            """Process a single task with error handling."""
            task_start = time.time()
            
            try:
                # Apply rate limiting if available
                if self.rate_limiter:
                    is_allowed, error_msg = self.rate_limiter.check_rate_limit(user_id)
                    if not is_allowed:
                        # Wait if rate limited
                        self.rate_limiter.wait_if_rate_limited(user_id, max_wait=60.0)
                
                # Execute task
                result = task_fn(task)
                execution_time = time.time() - task_start
                
                return BatchResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                )
            except Exception as e:
                execution_time = time.time() - task_start
                error_msg = str(e)
                logger.error(f"Error processing task {task_id}: {error_msg}")
                
                return BatchResult(
                    task_id=task_id,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                )
        
        # Process tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, task, task_id): (task, task_id)
                for task, task_id in zip(tasks, task_ids)
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                
                if show_progress:
                    logger.info(f"Batch progress: {completed}/{len(tasks)} tasks completed")
        
        # Calculate statistics
        total_time = time.time() - start_time
        completed_tasks = sum(1 for r in results if r.success)
        failed_tasks = len(results) - completed_tasks
        average_time = total_time / len(results) if results else 0.0
        
        stats = BatchStats(
            total_tasks=len(tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_time=total_time,
            average_time=average_time,
        )
        
        logger.info(
            f"Batch processing complete: {completed_tasks}/{len(tasks)} succeeded, "
            f"{failed_tasks} failed, {total_time:.2f}s total"
        )
        
        return results, stats
    
    async def process_batch_async(
        self,
        tasks: List[Any],
        task_fn: Callable,
        task_ids: Optional[List[str]] = None,
        user_id: str = "default",
        show_progress: bool = True,
    ) -> Tuple[List[BatchResult], BatchStats]:
        """Process a batch of tasks asynchronously.
        
        Args:
            tasks: List of task inputs
            task_fn: Async function to execute for each task
            task_ids: Optional list of task identifiers
            user_id: User identifier for rate limiting
            show_progress: Whether to show progress updates
            
        Returns:
            Tuple of (results, stats)
        """
        if not tasks:
            return [], BatchStats()
        
        start_time = time.time()
        results: List[BatchResult] = []
        
        # Generate task IDs if not provided
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(tasks))]
        
        if len(task_ids) != len(tasks):
            logger.warning("Task IDs length doesn't match tasks length, generating new IDs")
            task_ids = [f"task_{i}" for i in range(len(tasks))]
        
        async def process_single_task_async(task: Any, task_id: str) -> BatchResult:
            """Process a single task asynchronously with error handling."""
            task_start = time.time()
            
            try:
                # Apply rate limiting if available
                if self.rate_limiter:
                    is_allowed, error_msg = self.rate_limiter.check_rate_limit(user_id)
                    if not is_allowed:
                        # Wait if rate limited (async sleep)
                        await asyncio.sleep(1.0)
                        # Try again
                        is_allowed, error_msg = self.rate_limiter.check_rate_limit(user_id)
                        if not is_allowed:
                            await asyncio.sleep(5.0)
                
                # Execute task (assume it's async or can be awaited)
                if asyncio.iscoroutinefunction(task_fn):
                    result = await task_fn(task)
                else:
                    # Run sync function in executor
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running loop, create new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    result = await loop.run_in_executor(None, task_fn, task)
                
                execution_time = time.time() - task_start
                
                return BatchResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                )
            except Exception as e:
                execution_time = time.time() - task_start
                error_msg = str(e)
                logger.error(f"Error processing task {task_id}: {error_msg}")
                
                return BatchResult(
                    task_id=task_id,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                )
        
        # Process tasks concurrently
        tasks_to_run = [process_single_task_async(task, task_id) for task, task_id in zip(tasks, task_ids)]
        results = await asyncio.gather(*tasks_to_run)
        
        # Calculate statistics
        total_time = time.time() - start_time
        completed_tasks = sum(1 for r in results if r.success)
        failed_tasks = len(results) - completed_tasks
        average_time = total_time / len(results) if results else 0.0
        
        stats = BatchStats(
            total_tasks=len(tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_time=total_time,
            average_time=average_time,
        )
        
        logger.info(
            f"Async batch processing complete: {completed_tasks}/{len(tasks)} succeeded, "
            f"{failed_tasks} failed, {total_time:.2f}s total"
        )
        
        return results, stats
