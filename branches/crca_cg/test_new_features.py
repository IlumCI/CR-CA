#!/usr/bin/env python3
"""
Test script for new CorporateSwarm features:
- Automatic task generation
- CRCA evaluation
- Task decomposition
- Full execution monitoring
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from corposwarm import create_corporation

def test_task_generation():
    """Test automatic task generation"""
    print("\n" + "="*60)
    print("TEST 1: Automatic Task Generation")
    print("="*60)
    
    corp = create_corporation(
        name="TestCorp",
        verbose=True,
        budget_limit=50.0
    )
    
    print("\n📋 Generating tasks automatically...")
    tasks = corp.generate_tasks_automatically(max_tasks=3)
    
    print(f"\n✅ Generated {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    
    return corp, tasks

def test_crca_evaluation(corp, task):
    """Test CRCA evaluation of a task"""
    print("\n" + "="*60)
    print("TEST 2: CRCA Task Evaluation")
    print("="*60)
    
    print(f"\n🔍 Evaluating task: {task[:50]}...")
    evaluation = corp.evaluate_task_with_crca(task)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Overall Score: {evaluation.get('overall_score', 0.0):.2f}")
    print(f"  Positive Outcomes: {len(evaluation.get('positive_outcomes', []))}")
    print(f"  Negative Outcomes: {len(evaluation.get('negative_outcomes', []))}")
    
    if evaluation.get('negative_outcomes'):
        print(f"\n⚠️  Negative Outcomes Found:")
        for outcome in evaluation.get('negative_outcomes', [])[:3]:
            print(f"    - {outcome}")
    
    return evaluation

def test_task_refinement(corp, task, issues):
    """Test task refinement"""
    print("\n" + "="*60)
    print("TEST 3: Task Refinement")
    print("="*60)
    
    if not issues:
        print("\n✅ No issues to refine - task is already good!")
        return task
    
    print(f"\n🔧 Refining task to address {len(issues)} issues...")
    refined = corp.refine_task_with_llm(task, issues)
    
    print(f"\n✨ Refined Task:")
    print(f"  {refined}")
    
    return refined

def test_task_decomposition(corp, task):
    """Test task decomposition"""
    print("\n" + "="*60)
    print("TEST 4: Task Decomposition")
    print("="*60)
    
    print(f"\n🔨 Decomposing task: {task[:50]}...")
    subtasks = corp.decompose_task(task)
    
    print(f"\n📦 Decomposed into {len(subtasks)} sub-tasks:")
    for i, subtask in enumerate(subtasks, 1):
        print(f"\n  Sub-task {i}:")
        print(f"    Task: {subtask.get('task', 'N/A')[:60]}...")
        print(f"    Priority: {subtask.get('priority', 'N/A')}")
        print(f"    Effort: {subtask.get('estimated_effort', 'N/A')}")
        print(f"    Risk: {subtask.get('risk_level', 'N/A')}")
        print(f"    Dependencies: {subtask.get('dependencies', [])}")
    
    return subtasks

def test_subtask_resort(corp, subtasks):
    """Test sub-task re-sorting"""
    print("\n" + "="*60)
    print("TEST 5: Sub-task Re-sorting")
    print("="*60)
    
    print(f"\n🔄 Re-sorting {len(subtasks)} sub-tasks...")
    sorted_subtasks = corp.re_sort_subtasks(subtasks)
    
    print(f"\n✅ Re-sorted order:")
    for i, subtask in enumerate(sorted_subtasks, 1):
        deps = subtask.get('dependencies', [])
        deps_str = f" (depends on: {deps})" if deps else " (no dependencies)"
        print(f"  {i}. Priority {subtask.get('priority', 'N/A')}, Risk {subtask.get('risk_level', 'N/A')}{deps_str}")
    
    return sorted_subtasks

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CorporateSwarm New Features Test Suite")
    print("="*60)
    
    try:
        # Test 1: Task Generation
        corp, tasks = test_task_generation()
        
        if not tasks:
            print("\n❌ No tasks generated - cannot continue tests")
            return
        
        # Test 2: CRCA Evaluation
        task = tasks[0]
        evaluation = test_crca_evaluation(corp, task)
        
        # Test 3: Task Refinement (if needed)
        negative_outcomes = evaluation.get('negative_outcomes', [])
        if negative_outcomes:
            refined_task = test_task_refinement(corp, task, negative_outcomes[:3])
            task = refined_task
        
        # Test 4: Task Decomposition
        subtasks = test_task_decomposition(corp, task)
        
        # Test 5: Sub-task Re-sorting
        if len(subtasks) > 1:
            sorted_subtasks = test_subtask_resort(corp, subtasks)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📝 Summary:")
        print(f"  - Generated {len(tasks)} tasks")
        print(f"  - Evaluated task with CRCA (score: {evaluation.get('overall_score', 0.0):.2f})")
        if negative_outcomes:
            print(f"  - Refined task to address {len(negative_outcomes)} issues")
        print(f"  - Decomposed task into {len(subtasks)} sub-tasks")
        if len(subtasks) > 1:
            print(f"  - Re-sorted sub-tasks for optimal execution")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

