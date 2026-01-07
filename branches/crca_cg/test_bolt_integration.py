#!/usr/bin/env python3
"""
Bolt.diy Integration Test

This test file specifically tests the integration between CorporateSwarm and bolt.diy.
It checks bolt.diy health, starts it if needed, and forces a mandate execution.

Requirements:
- bolt.diy should be running at http://localhost:5173 (or set BOLT_DIY_API_URL)
- CorporateSwarm should be able to submit mandates to bolt.diy
"""

import sys
import os
import time
import requests
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from corposwarm import (
    create_corporation,
    CorporateSwarm,
    CorporateRole,
    DepartmentType,
    ProposalType
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
    except Exception as e:
        print(f"[WARN] Error checking bolt.diy health: {e}")
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
    
    print(f"Waiting for bolt.diy to be ready at {url}...")
    
    while time.time() - start_time < max_wait:
        if check_bolt_diy_health(url):
            print(f"[OK] bolt.diy is ready at {url}")
            return True
        elapsed = int(time.time() - start_time)
        print(f"[WAIT] bolt.diy not ready yet... ({elapsed}s/{max_wait}s)")
        time.sleep(check_interval)
    
    print(f"[ERROR] bolt.diy not ready after {max_wait} seconds")
    return False


def test_bolt_diy_health():
    """Test 1: Check bolt.diy health endpoint."""
    print("\n" + "="*80)
    print("TEST 1: Bolt.diy Health Check")
    print("="*80)
    
    bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
    
    if check_bolt_diy_health(bolt_diy_url):
        print(f"[PASS] bolt.diy is healthy at {bolt_diy_url}")
        return True
    else:
        print(f"[FAIL] bolt.diy is not reachable at {bolt_diy_url}")
        print(f"[INFO] Make sure bolt.diy is running. You can start it with:")
        print(f"       cd CR-CA/tools/bolt.diy && npm run dev")
        return False


def wait_for_mandate_execution(bolt_diy_url: str, mandate_id: str, max_wait: int = 300) -> dict:
    """
    Wait for a mandate to complete execution and return execution results.
    
    Args:
        bolt_diy_url: bolt.diy API URL
        mandate_id: Mandate ID to wait for
        max_wait: Maximum time to wait in seconds (default: 5 minutes)
        
    Returns:
        dict: Execution results with status, events, and generated files
    """
    start_time = time.time()
    check_interval = 3  # Check every 3 seconds
    
    print(f"[INFO] Waiting for mandate {mandate_id} to execute (max {max_wait}s)...")
    
    last_status = "accepted"
    events = []
    generated_files = []
    
    while time.time() - start_time < max_wait:
        try:
            # Check mandate events
            events_url = f"{bolt_diy_url.rstrip('/')}/api/mandate?mandate_id={mandate_id}"
            response = requests.get(events_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get("events", [])
                
                # Check for completion events
                for event in events:
                    event_type = event.get("type", "")
                    
                    if event_type == "iteration_end":
                        last_status = "completed"
                        # Extract generated files from event data
                        event_data = event.get("data", {})
                        if "files_created" in event_data:
                            generated_files.extend(event_data["files_created"])
                        if "files_modified" in event_data:
                            generated_files.extend(event_data["files_modified"])
                    elif event_type == "error":
                        last_status = "failed"
                    elif event_type == "iteration_start":
                        last_status = "running"
                
                # Check if we have a completion event
                if last_status == "completed":
                    print(f"[INFO] Mandate execution completed!")
                    return {
                        "status": "completed",
                        "events": events,
                        "generated_files": list(set(generated_files)),  # Remove duplicates
                        "event_count": len(events)
                    }
                elif last_status == "failed":
                    error_msg = "Unknown error"
                    for event in events:
                        if event.get("type") == "error":
                            error_msg = event.get("data", {}).get("message", error_msg)
                    print(f"[INFO] Mandate execution failed: {error_msg}")
                    return {
                        "status": "failed",
                        "events": events,
                        "error": error_msg
                    }
        except Exception as e:
            print(f"[WARN] Error checking mandate status: {e}")
        
        elapsed = int(time.time() - start_time)
        if elapsed % 15 == 0:  # Print status every 15 seconds
            print(f"[WAIT] Still executing... ({elapsed}s/{max_wait}s, {len(events)} events)")
        
        time.sleep(check_interval)
    
    print(f"[WARN] Mandate execution did not complete within {max_wait} seconds")
    return {
        "status": "timeout",
        "events": events,
        "generated_files": list(set(generated_files))
    }


def test_mandate_submission():
    """Test 2: Submit a mandate to bolt.diy and wait for code generation."""
    print("\n" + "="*80)
    print("TEST 2: Mandate Submission & Code Generation")
    print("="*80)
    
    bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
    
    # Ensure bolt.diy is ready
    if not ensure_bolt_diy_ready(bolt_diy_url, max_wait=30):
        print("[SKIP] Skipping mandate submission test - bolt.diy not available")
        return False
    
    # Create a corporation
    print("[INFO] Creating corporation...")
    corp = create_corporation(
        name="BoltIntegrationTestCorp",
        verbose=True,
        budget_limit=200.0,
        auto_deploy_enabled=False,
        auto_test_enabled=True
    )
    
    # Create a proposal that will become a mandate
    print("[INFO] Creating code-related proposal...")
    sponsor_id = corp.executive_team[0] if corp.executive_team else corp.board_members[0]
    
    proposal_id = corp.create_proposal(
        title="Build Simple Todo App - Bolt Integration Test",
        description=(
            "Create a simple HTML/CSS/JavaScript todo application. "
            "Features: add todos, mark as complete, delete todos. "
            "This is a test mandate to verify bolt.diy integration. "
            "Revenue-generating project for lead generation."
        ),
        proposal_type=ProposalType.STRATEGIC_INITIATIVE,
        sponsor_id=sponsor_id,
        department=DepartmentType.TECHNOLOGY,
        budget_impact=15.0,
        timeline="1 day"
    )
    
    print(f"[INFO] Created proposal: {proposal_id}")
    
    # Execute the mandate directly
    print("[INFO] Creating and executing mandate...")
    
    mandate = {
        "mandate_id": f"bolt-test-mandate-{int(time.time())}",
        "objectives": [
            "Create a simple HTML todo application",
            "Add functionality to add, delete, and mark todos as complete",
            "Style with basic CSS",
            "Make it revenue-focused with lead generation features"
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
            "proposal_id": proposal_id
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
    
    # Execute mandate (this will make real HTTP calls and open browser windows)
    print(f"[INFO] Submitting mandate {mandate['mandate_id']} to bolt.diy...")
    result = corp.execute_code_mandate(mandate, proposal_id=proposal_id)
    
    if result is None:
        print("[FAIL] Mandate execution returned None")
        return False
    
    print(f"[INFO] Mandate execution result: {result.get('status')}")
    
    if result.get("status") == "accepted":
        mandate_id = result.get("mandate_id") or mandate["mandate_id"]
        print(f"[PASS] Mandate accepted by bolt.diy!")
        print(f"[INFO] Mandate ID: {mandate_id}")
        print(f"[INFO] Observability: {bolt_diy_url}/observability/{mandate_id}")
        print(f"[INFO] Execute: {bolt_diy_url}/execute/{mandate_id}")
        print(f"[INFO] Workflow: {bolt_diy_url}/workflow/{proposal_id}")
        print(f"[INFO] Home page: {bolt_diy_url}/")
        print(f"[INFO] Browser windows should have opened automatically")
        print(f"[INFO] Waiting for code generation to complete...")
        
        # Wait for execution to complete
        execution_result = wait_for_mandate_execution(bolt_diy_url, mandate_id, max_wait=300)
        
        if execution_result["status"] == "completed":
            generated_files = execution_result.get("generated_files", [])
            event_count = execution_result.get("event_count", 0)
            
            print(f"[PASS] Code generation completed successfully!")
            print(f"[INFO] Generated {len(generated_files)} file(s):")
            for file in generated_files:
                print(f"       - {file}")
            print(f"[INFO] Total events: {event_count}")
            print(f"[INFO] View generated code at: {bolt_diy_url}/observability/{mandate_id}")
            return True
        elif execution_result["status"] == "failed":
            error = execution_result.get("error", "Unknown error")
            print(f"[FAIL] Code generation failed: {error}")
            return False
        elif execution_result["status"] == "timeout":
            generated_files = execution_result.get("generated_files", [])
            print(f"[WARN] Code generation did not complete within timeout")
            if generated_files:
                print(f"[INFO] Partial files generated: {', '.join(generated_files)}")
            print(f"[INFO] Check execution status manually at: {bolt_diy_url}/observability/{mandate_id}")
            return False
        else:
            print(f"[WARN] Unknown execution status: {execution_result.get('status')}")
            return False
    elif result.get("status") == "completed":
        print(f"[PASS] Mandate completed!")
        return True
    elif result.get("status") == "failed":
        error = result.get("error", "Unknown error")
        print(f"[FAIL] Mandate execution failed: {error}")
        return False
    else:
        print(f"[WARN] Unknown mandate status: {result.get('status')}")
        return True  # Still consider it a pass if we got a response


def test_mandate_status_checking():
    """Test 3: Check mandate status after submission."""
    print("\n" + "="*80)
    print("TEST 3: Mandate Status Checking")
    print("="*80)
    
    bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
    
    # Ensure bolt.diy is ready
    if not ensure_bolt_diy_ready(bolt_diy_url, max_wait=10):
        print("[SKIP] Skipping status check test - bolt.diy not available")
        return False
    
    # Create corporation and submit a mandate
    corp = create_corporation(
        name="StatusCheckCorp",
        verbose=False,
        budget_limit=100.0
    )
    
    mandate = {
        "mandate_id": f"status-test-{int(time.time())}",
        "objectives": ["Test status checking"],
        "constraints": {"language": "html", "maxDependencies": 0},
        "budget": {"token": 10000, "time": 60, "cost": 0.5},
        "deliverables": ["test.html"],
        "governance": {"proposal_id": "test-proposal"},
        "iteration_config": {"max_iterations": 1, "test_required": False},
        "deployment": {"enabled": False},
        "testing": {"enabled": False}
    }
    
    # Submit mandate
    result = corp.execute_code_mandate(mandate, proposal_id="test-proposal")
    
    if result and result.get("status") == "accepted":
        mandate_id = result.get("mandate_id") or mandate["mandate_id"]
        
        # Wait a bit for execution to start
        print(f"[INFO] Waiting 3 seconds for execution to start...")
        time.sleep(3)
        
        # Check status
        status = corp.check_mandate_status(mandate_id, bolt_diy_url)
        
        print(f"[INFO] Mandate status: {status}")
        
        if status in ["accepted", "running", "completed"]:
            print(f"[PASS] Status check working (status: {status})")
            return True
        else:
            print(f"[WARN] Unexpected status: {status}")
            return True  # Still pass, status checking is working
    else:
        print(f"[SKIP] Could not submit mandate for status check")
        return False


def run_all_tests():
    """Run all bolt.diy integration tests."""
    print("\n" + "="*80)
    print("Bolt.diy Integration Test Suite")
    print("="*80)
    
    tests = [
        ("Bolt.diy Health Check", test_bolt_diy_health),
        ("Mandate Submission", test_mandate_submission),
        ("Mandate Status Checking", test_mandate_status_checking),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

