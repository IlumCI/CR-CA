"""
Flask application for CRCA Web UI.

Provides a simple REST API and serves the web interface.
"""

import json
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template, session
from loguru import logger

from webui.config import AGENTS, get_default_agent_key, get_agent_info, list_agents

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "crca-webui-secret-key-change-in-production"  # Change in production

# In-memory agent storage (session-based)
# In production, consider using Redis or database
_agent_instances: Dict[str, Any] = {}


def get_agent_instance(agent_key: str) -> Optional[Any]:
    """
    Get or create an agent instance.
    
    Args:
        agent_key: Agent registry key
        
    Returns:
        Agent instance or None if creation fails
    """
    # Check if we already have an instance for this session
    session_id = session.get("session_id", "default")
    cache_key = f"{session_id}_{agent_key}"
    
    if cache_key in _agent_instances:
        return _agent_instances[cache_key]
    
    # Get agent info
    agent_info = get_agent_info(agent_key)
    if not agent_info:
        logger.error(f"Agent '{agent_key}' not found in registry")
        return None
    
    # Create new instance
    agent_class = agent_info["class"]
    try:
        if agent_key == "general":
            # GeneralAgent can be initialized with no args (uses defaults)
            instance = agent_class()
        elif agent_key == "hybrid":
            # HybridAgent can be initialized with no args (uses defaults)
            instance = agent_class()
        else:
            # Generic initialization for other agents
            instance = agent_class()
        
        _agent_instances[cache_key] = instance
        logger.info(f"Created {agent_info['name']} instance for session {session_id}")
        return instance
    
    except Exception as e:
        logger.error(f"Failed to create agent instance: {e}")
        return None


@app.route("/")
def index() -> str:
    """Serve the main web UI page."""
    return render_template("index.html")


@app.route("/api/agents", methods=["GET"])
def api_agents() -> Any:
    """
    Get list of available agents.
    
    Returns:
        JSON response with agent list
    """
    agents = list_agents()
    
    # Format for frontend
    agent_list = []
    for key, info in agents.items():
        agent_list.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "requires_llm": info.get("requires_llm", True),
            "default": info.get("default", False),
        })
    
    return jsonify({
        "agents": agent_list,
        "default": get_default_agent_key(),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    """
    Handle chat message and return agent response.
    
    Request body:
        {
            "message": str,
            "agent_key": str (optional, uses default if not provided)
        }
    
    Returns:
        JSON response with agent reply
    """
    # Initialize session if needed
    if "session_id" not in session:
        import uuid
        session["session_id"] = str(uuid.uuid4())
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get agent key (use default if not provided)
        agent_key = data.get("agent_key") or get_default_agent_key()
        if not agent_key:
            return jsonify({"error": "No agents available"}), 500
        
        # Get agent instance
        agent = get_agent_instance(agent_key)
        if not agent:
            return jsonify({"error": f"Failed to initialize agent '{agent_key}'"}), 500
        
        # Get agent info for response
        agent_info = get_agent_info(agent_key)
        agent_name = agent_info["name"] if agent_info else agent_key
        
        # Run agent
        try:
            response = agent.run(message)
            
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)
            
            return jsonify({
                "response": response,
                "agent_type": agent_key,
                "agent_name": agent_name,
            })
        
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return jsonify({
                "error": f"Agent execution failed: {str(e)}",
                "agent_type": agent_key,
            }), 500
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/switch", methods=["POST"])
def api_switch() -> Any:
    """
    Switch to a different agent.
    
    Request body:
        {
            "agent_key": str
        }
    
    Returns:
        JSON response confirming switch
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        agent_key = data.get("agent_key")
        if not agent_key:
            return jsonify({"error": "agent_key is required"}), 400
        
        # Verify agent exists
        agent_info = get_agent_info(agent_key)
        if not agent_info:
            return jsonify({"error": f"Agent '{agent_key}' not found"}), 404
        
        # Clear cached instance for this session to force recreation
        session_id = session.get("session_id", "default")
        cache_key = f"{session_id}_{agent_key}"
        if cache_key in _agent_instances:
            del _agent_instances[cache_key]
        
        return jsonify({
            "success": True,
            "agent_key": agent_key,
            "agent_name": agent_info["name"],
        })
    
    except Exception as e:
        logger.error(f"Error switching agent: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error: Any) -> Any:
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error: Any) -> Any:
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Run development server
    logger.info("Starting CRCA Web UI server...")
    logger.info(f"Available agents: {list(AGENTS.keys())}")
    app.run(host="0.0.0.0", port=5000, debug=True)
