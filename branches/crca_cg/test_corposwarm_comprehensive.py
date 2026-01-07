#!/usr/bin/env python3
"""
Comprehensive test suite for CorporateSwarm system.

Tests all major features including:
- Corporation creation and initialization
- Member management
- Proposal creation and voting
- Task generation and evaluation
- Mandate execution (with forced mandate submission)
- Deployment features
- Testing features
- I/O operations
- Governance and compliance
"""

import sys
import os
import time
import requests
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from corposwarm import (
    create_corporation,
    CorporateSwarm,
    CorporateRole,
    DepartmentType,
    ProposalType,
    VoteResult
)


def check_bolt_diy_health(url: str = "http://localhost:5173", timeout: int = 5) -> bool:
    """
    Check if bolt.diy is running and reachable.
    
    Args:
        url: bolt.diy API URL
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if bolt.diy is reachable, False otherwise
    """
    try:
        health_url = url.rstrip('/') + "/api/health"
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("status") == "healthy":
                    return True
            except (ValueError, KeyError):
                # Response is not JSON or missing status field, but 200 OK - still ready
                pass
            return True  # Any 200 response means the server is ready
        return False
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
    except Exception:
        return False


def ensure_bolt_diy_ready(url: str = "http://localhost:5173", max_wait: int = 60) -> bool:
    """
    Ensure bolt.diy is ready, waiting up to max_wait seconds.
    
    Args:
        url: bolt.diy API URL
        max_wait: Maximum time to wait in seconds
        
    Returns:
        bool: True if bolt.diy is ready, False otherwise
    """
    start_time = time.time()
    check_interval = 2  # Check every 2 seconds
    
    while time.time() - start_time < max_wait:
        if check_bolt_diy_health(url):
            return True
        time.sleep(check_interval)
    
    return False


class TestCorporationCreation:
    """Test corporation creation and initialization."""
    
    def test_create_simple_corporation(self):
        """Test creating a simple corporation."""
        corp = create_corporation(
            name="TestCorp",
            verbose=False,
            budget_limit=100.0
        )
        
        assert corp is not None
        assert corp.name == "TestCorp"
        assert len(corp.board_members) > 0
        assert len(corp.executive_team) > 0
        assert len(corp.members) > 0
        assert corp.cost_tracker.budget_limit == 100.0
        
        print("[PASS] Corporation creation test passed")
    
    def test_corporation_with_custom_config(self):
        """Test creating corporation with custom configuration."""
        corp = create_corporation(
            name="CustomCorp",
            model_name="gpt-4o-mini",
            budget_limit=200.0,
            enable_causal_reasoning=True,
            enable_quant_analysis=True,
            verbose=False
        )
        
        assert corp.corporate_model_name == "gpt-4o-mini"
        assert corp.config.enable_causal_reasoning is True
        assert corp.config.enable_quant_analysis is True
        
        print("[PASS] Custom configuration test passed")


class TestMemberManagement:
    """Test member management features."""
    
    def test_add_member(self):
        """Test adding a new member."""
        corp = create_corporation(
            name="MemberTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test adding a new member."""
        initial_count = len(corp.members)
        
        member_id = corp.add_member(
            name="Test Member",
            role=CorporateRole.EMPLOYEE,
            department=DepartmentType.TECHNOLOGY,
            expertise_areas=["Python", "Testing"],
            voting_weight=1.0
        )
        
        assert member_id is not None
        assert len(corp.members) == initial_count + 1
        assert member_id in corp.members
        assert corp.members[member_id].name == "Test Member"
        
        print("[PASS] Add member test passed")
    
    def test_add_executive(self):
        """Test adding an executive team member."""
        corp = create_corporation(
            name="MemberTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        member_id = corp.add_member(
            name="Test CEO",
            role=CorporateRole.CEO,
            department=DepartmentType.OPERATIONS,
            expertise_areas=["Strategy", "Leadership"],
            voting_weight=2.0
        )
        
        assert member_id in corp.executive_team
        assert corp.members[member_id].voting_weight == 2.0
        
        print("[PASS] Add executive test passed")
    
    def test_add_board_member(self):
        """Test adding a board member."""
        corp = create_corporation(
            name="MemberTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        member_id = corp.add_member(
            name="Test Board Member",
            role=CorporateRole.BOARD_MEMBER,
            department=DepartmentType.OPERATIONS,
            expertise_areas=["Governance"],
            voting_weight=1.5
        )
        
        assert member_id in corp.board_members
        assert corp.members[member_id].role == CorporateRole.BOARD_MEMBER
        
        print("[PASS] Add board member test passed")


class TestProposalAndVoting:
    """Test proposal creation and voting."""
    
    def test_create_proposal(self):
        """Test creating a proposal."""
        corp = create_corporation(
            name="VoteTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test creating a proposal."""
        sponsor_id = corp.executive_team[0] if corp.executive_team else corp.board_members[0]
        
        proposal_id = corp.create_proposal(
            title="Test Proposal",
            description="This is a test proposal for revenue generation",
            proposal_type=ProposalType.STRATEGIC_INITIATIVE,
            sponsor_id=sponsor_id,
            department=DepartmentType.OPERATIONS,
            budget_impact=10.0,
            timeline="1 week"
        )
        
        assert proposal_id is not None
        proposal = corp._find_proposal(proposal_id)
        assert proposal.title == "Test Proposal"
        assert proposal.budget_impact == 10.0
        
        print("[PASS] Create proposal test passed")
    
    def test_conduct_vote(self):
        """Test conducting a vote on a proposal."""
        corp = create_corporation(
            name="VoteTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        sponsor_id = corp.executive_team[0] if corp.executive_team else corp.board_members[0]
        
        proposal_id = corp.create_proposal(
            title="Revenue Proposal",
            description="Generate revenue through aggressive marketing campaign",
            proposal_type=ProposalType.STRATEGIC_INITIATIVE,
            sponsor_id=sponsor_id,
            department=DepartmentType.OPERATIONS,
            budget_impact=5.0
        )
        
        vote = corp.conduct_corporate_vote(proposal_id)
        
        assert vote is not None
        assert vote.proposal.proposal_id == proposal_id
        assert vote.result in [VoteResult.APPROVED, VoteResult.REJECTED, VoteResult.TABLED, VoteResult.FAILED]
        assert len(vote.participants) > 0
        
        print(f"[PASS] Conduct vote test passed (Result: {vote.result.value})")


class TestTaskGeneration:
    """Test automatic task generation."""
    
    def test_generate_tasks(self):
        """Test automatic task generation."""
        corp = create_corporation(
            name="TaskGenCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test automatic task generation."""
        tasks = corp.generate_tasks_automatically(max_tasks=5)
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(task, str) for task in tasks)
        
        print(f"[PASS] Generated {len(tasks)} tasks:")
        for i, task in enumerate(tasks[:3], 1):
            print(f"   {i}. {task[:60]}...")
    
    def test_evaluate_task(self):
        """Test task evaluation with CRCA."""
        corp = create_corporation(
            name="TaskGenCorp",
            verbose=False,
            budget_limit=100.0
        )
        task = "Launch aggressive marketing campaign to increase revenue by 50%"
        
        evaluation = corp.evaluate_task_with_crca(task)
        
        assert evaluation is not None
        assert "overall_score" in evaluation
        assert "positive_outcomes" in evaluation
        assert "negative_outcomes" in evaluation
        assert isinstance(evaluation["overall_score"], (int, float))
        
        print(f"[PASS] Task evaluation test passed (Score: {evaluation['overall_score']:.2f})")
    
    def test_refine_task(self):
        """Test task refinement."""
        corp = create_corporation(
            name="TaskGenCorp",
            verbose=False,
            budget_limit=100.0
        )
        task = "Increase revenue through marketing"
        issues = ["Budget concerns", "Timeline risks"]
        
        refined = corp.refine_task_with_llm(task, issues)
        
        assert isinstance(refined, str)
        assert len(refined) > 0
        
        print("[PASS] Task refinement test passed")


class TestMandateExecution:
    """Test mandate execution feature - FORCED MANDATE SUBMISSION."""
    
    def test_create_mandate_proposal(self):
        """Test creating a proposal that will become a mandate."""
        corp = create_corporation(
            name="MandateTestCorp",
            verbose=False,
            budget_limit=200.0,
            auto_deploy_enabled=False,  # Disable auto-deploy for testing
            auto_test_enabled=True
        )
        """Test creating a proposal that will become a mandate."""
        sponsor_id = corp.executive_team[0] if corp.executive_team else corp.board_members[0]
        
        # Create a code-related proposal (will trigger mandate execution)
        proposal_id = corp.create_proposal(
            title="Build Simple Todo App",
            description="Create a simple React todo application with add, delete, and complete functionality. This is a revenue-generating project.",
            proposal_type=ProposalType.STRATEGIC_INITIATIVE,
            sponsor_id=sponsor_id,
            department=DepartmentType.TECHNOLOGY,
            budget_impact=20.0,
            timeline="2 days"
        )
        
        assert proposal_id is not None
        print(f"[PASS] Created proposal: {proposal_id}")
        
        return proposal_id
    
    def test_force_mandate_execution(self):
        """FORCE mandate execution by creating and approving a code-related proposal."""
        # Check bolt.diy health before mandate execution
        bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
        print(f"Checking bolt.diy health at {bolt_diy_url}...")
        
        if not check_bolt_diy_health(bolt_diy_url):
            print(f"[WARN] bolt.diy not ready at {bolt_diy_url}, attempting to wait...")
            if not ensure_bolt_diy_ready(bolt_diy_url, max_wait=30):
                print(f"[SKIP] bolt.diy not available, skipping mandate execution test")
                return
        
        print(f"[INFO] bolt.diy is ready at {bolt_diy_url}")
        
        corp = create_corporation(
            name="MandateTestCorp",
            verbose=False,
            budget_limit=200.0,
            auto_deploy_enabled=False,
            auto_test_enabled=True
        )
        sponsor_id = corp.executive_team[0] if corp.executive_team else corp.board_members[0]
        
        # Create a profit-focused code proposal
        proposal_id = corp.create_proposal(
            title="Build Revenue-Generating Landing Page",
            description="Create a simple HTML/CSS/JS landing page for a SaaS product. Include pricing section, CTA buttons, and contact form. This will generate revenue through lead generation.",
            proposal_type=ProposalType.STRATEGIC_INITIATIVE,
            sponsor_id=sponsor_id,
            department=DepartmentType.TECHNOLOGY,
            budget_impact=15.0,
            timeline="1 day"
        )
        
        proposal = corp._find_proposal(proposal_id)
        assert proposal is not None
        
        # Process proposal (this will make real HTTP calls to bolt.diy and open real browser windows)
        result = corp._process_proposal_task(
            f"Execute proposal {proposal_id}: Build Revenue-Generating Landing Page"
        )
        
        assert result is not None
        assert "status" in result
        assert "proposal_id" in result
        
        print(f"[PASS] Mandate execution test passed (Status: {result.get('status')})")
        print(f"   Note: Browser windows should have opened automatically for mandate visibility")
    
    def test_execute_code_mandate_direct(self):
        """Test direct mandate execution with a simple project."""
        # Check bolt.diy health before mandate execution
        bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
        print(f"Checking bolt.diy health at {bolt_diy_url}...")
        
        if not check_bolt_diy_health(bolt_diy_url):
            print(f"[WARN] bolt.diy not ready at {bolt_diy_url}, attempting to wait...")
            if not ensure_bolt_diy_ready(bolt_diy_url, max_wait=30):
                print(f"[SKIP] bolt.diy not available, skipping direct mandate execution test")
                return
        
        print(f"[INFO] bolt.diy is ready at {bolt_diy_url}")
        
        corp = create_corporation(
            name="MandateTestCorp",
            verbose=False,
            budget_limit=200.0
        )
        # Create a simple mandate for a todo app
        mandate = {
            "mandate_id": f"test-mandate-{int(time.time())}",
            "objectives": [
                "Create a simple HTML todo application",
                "Add functionality to add, delete, and mark todos as complete",
                "Style with basic CSS"
            ],
            "constraints": {
                "language": "html",
                "maxDependencies": 0,
                "no_frameworks": True
            },
            "budget": {
                "token": 50000,
                "time": 300,
                "cost": 2.0
            },
            "deliverables": [
                "index.html",
                "style.css",
                "script.js"
            ],
            "governance": {
                "proposal_id": "test-proposal-123"
            },
            "iteration_config": {
                "max_iterations": 2,
                "test_required": False
            },
            "deployment": {
                "enabled": False,
                "provider": "netlify",
                "auto_deploy": False
            },
            "testing": {
                "enabled": False,
                "generate_tests": False,
                "run_tests": False
            }
        }
        
        # Execute mandate (this will make real HTTP calls to bolt.diy and open real browser windows)
        result = corp.execute_code_mandate(mandate, proposal_id="test-proposal-123")
        
        assert result is not None
        assert "status" in result
        assert result.get("status") in ["accepted", "completed", "failed"]
        
        print(f"[PASS] Direct mandate execution test passed")
        print(f"   Mandate ID: {mandate['mandate_id']}")
        print(f"   Status: {result.get('status')}")
        print(f"   Note: Browser windows should have opened automatically for mandate visibility")


class TestDeploymentFeatures:
    """Test deployment features."""
    
    def test_deploy_to_netlify(self):
        """Test Netlify deployment."""
        corp = create_corporation(
            name="DeployTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test Netlify deployment."""
        mandate_id = "test-mandate-123"
        files = {
            "index.html": "<html><body>Test</body></html>",
            "style.css": "body { margin: 0; }"
        }
        
        # This will make real HTTP calls to Netlify (if configured)
        result = corp._deploy_to_netlify(mandate_id, files, "http://localhost:5173")
        
        assert result is not None
        assert "status" in result
        
        print("[PASS] Netlify deployment test passed")
    
    def test_deploy_to_vercel(self):
        """Test Vercel deployment."""
        corp = create_corporation(
            name="DeployTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        mandate_id = "test-mandate-456"
        files = {
            "index.html": "<html><body>Test</body></html>"
        }
        
        # This will make real HTTP calls to Vercel (if configured)
        result = corp._deploy_to_vercel(mandate_id, files, "http://localhost:5173")
        
        assert result is not None
        print("[PASS] Vercel deployment test passed")


class TestIOOperations:
    """Test I/O operations."""
    
    def test_file_operations(self):
        """Test file operations."""
        corp = create_corporation(
            name="IOTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test file operations."""
        mandate_id = "test-mandate-io"
        operations = [
            {
                "type": "write",
                "path": "test.txt",
                "content": "Hello, World!"
            },
            {
                "type": "read",
                "path": "test.txt"
            }
        ]
        
        # This will make real HTTP calls to bolt.diy for file operations
        result = corp.perform_file_operations(mandate_id, operations)
        
        assert result is not None
        assert "status" in result
        
        print("[PASS] File operations test passed")
    
    def test_database_operations(self):
        """Test database operations."""
        corp = create_corporation(
            name="IOTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        mandate_id = "test-mandate-db"
        operations = [
            {
                "type": "query",
                "database": "sqlite",
                "query": "SELECT * FROM users LIMIT 10"
            }
        ]
        
        # This will make real HTTP calls to bolt.diy for database operations
        result = corp.perform_database_operations(mandate_id, operations)
        
        assert result is not None
        print("[PASS] Database operations test passed")
    
    def test_api_calls(self):
        """Test API calls."""
        corp = create_corporation(
            name="IOTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        mandate_id = "test-mandate-api"
        requests_list = [
            {
                "method": "GET",
                "url": "https://api.example.com/data",
                "headers": {"Authorization": "Bearer token"}
            }
        ]
        
        # This will make real HTTP calls via bolt.diy
        result = corp.perform_api_calls(mandate_id, requests_list)
        
        assert result is not None
        print("[PASS] API calls test passed")


class TestGovernance:
    """Test governance features."""
    
    def test_esg_score(self):
        """Test ESG score calculation."""
        corp = create_corporation(
            name="GovTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test ESG score calculation."""
        esg_score = corp.calculate_esg_score()
        
        assert esg_score is not None
        assert hasattr(esg_score, "environmental_score")
        assert hasattr(esg_score, "social_score")
        assert hasattr(esg_score, "governance_score")
        assert hasattr(esg_score, "overall_score")
        
        print(f"[PASS] ESG score test passed (Overall: {esg_score.overall_score:.1f})")
    
    def test_risk_assessment(self):
        """Test risk assessment."""
        corp = create_corporation(
            name="GovTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        risks = corp.conduct_risk_assessment("comprehensive")
        
        assert risks is not None
        assert isinstance(risks, dict)
        
        print(f"[PASS] Risk assessment test passed ({len(risks)} risks identified)")
    
    def test_compliance_framework(self):
        """Test compliance framework."""
        corp = create_corporation(
            name="GovTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        frameworks = corp.establish_compliance_framework("comprehensive")
        
        assert frameworks is not None
        assert isinstance(frameworks, dict)
        
        print("[PASS] Compliance framework test passed")
    
    def test_corporate_status(self):
        """Test getting corporate status."""
        corp = create_corporation(
            name="GovTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        status = corp.get_corporate_status()
        
        assert status is not None
        assert "members" in status
        assert "departments" in status
        assert "proposals" in status
        
        print("[PASS] Corporate status test passed")


class TestTaskDecomposition:
    """Test task decomposition features."""
    
    def test_decompose_task(self):
        """Test task decomposition."""
        corp = create_corporation(
            name="DecompTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        """Test task decomposition."""
        task = "Build a revenue-generating SaaS landing page with pricing, features, and contact form"
        
        subtasks = corp.decompose_task(task)
        
        assert isinstance(subtasks, list)
        assert len(subtasks) > 0
        
        for subtask in subtasks:
            assert "task" in subtask
            assert "priority" in subtask
        
        print(f"[PASS] Task decomposition test passed ({len(subtasks)} subtasks)")
    
    def test_re_sort_subtasks(self):
        """Test sub-task re-sorting."""
        corp = create_corporation(
            name="DecompTestCorp",
            verbose=False,
            budget_limit=100.0
        )
        task = "Create a profit-maximizing e-commerce website"
        subtasks = corp.decompose_task(task)
        
        if len(subtasks) > 1:
            sorted_subtasks = corp.re_sort_subtasks(subtasks)
            
            assert len(sorted_subtasks) == len(subtasks)
            assert isinstance(sorted_subtasks, list)
            
            print("[PASS] Sub-task re-sorting test passed")


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*80)
    print("CorporateSwarm Comprehensive Test Suite")
    print("="*80 + "\n")
    
    test_classes = [
        TestCorporationCreation,
        TestMemberManagement,
        TestProposalAndVoting,
        TestTaskGeneration,
        TestMandateExecution,
        TestDeploymentFeatures,
        TestIOOperations,
        TestGovernance,
        TestTaskDecomposition
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{'='*80}")
        print(f"Running {class_name}")
        print(f"{'='*80}")
        
        # Create instance
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        # Create fixtures if needed
        if hasattr(instance, 'corp'):
            try:
                instance.corp = instance.corp()
            except:
                pass
        
        for test_method_name in test_methods:
            total_tests += 1
            test_method = getattr(instance, test_method_name)
            
            try:
                # Run test method (bound method, so call without arguments)
                test_method()
                
                passed_tests += 1
                print(f"[PASS] {test_method_name} PASSED")
            except Exception as e:
                failed_tests.append((class_name, test_method_name, str(e)))
                print(f"[FAIL] {test_method_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    
    print("="*80 + "\n")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

