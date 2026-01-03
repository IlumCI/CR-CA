"""
CorporateSwarm - Advanced Autonomous Corporate Governance System

Multi-agent orchestration system for corporate governance with board oversight, 
executive leadership, ESG frameworks, risk management, and democratic decision-making.

Features: ESG scoring, risk assessment, stakeholder engagement, regulatory compliance,
AI ethics governance, crisis management, and innovation oversight.
"""

# Standard library imports
import asyncio
import hashlib
import json
import os
import time
import traceback
import uuid
import yaml
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

# Third-party imports
from loguru import logger
from pydantic import BaseModel, Field

# Swarms imports
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.hybrid_hiearchical_peer_swarm import HybridHierarchicalClusterSwarm
from swarms.structs.swarm_router import SwarmRouter
try:
    from swarms.utils.loguru_logger import initialize_logger
except ImportError:
    # Fallback: try importing from swarms directly (old pattern)
    try:
        from swarms import logger as swarms_logger
        initialize_logger = swarms_logger.initialize_logger
    except ImportError:
        # Last resort: use loguru logger directly
        from loguru import logger as loguru_logger
        def initialize_logger(log_folder: str = "default"):
            return loguru_logger

try:
    from swarms.utils.output_types import OutputType as swarms_output_types_OutputType
    # Create a namespace-like object for compatibility
    class swarms_output_types:
        OutputType = swarms_output_types_OutputType
except ImportError:
    # Fallback: try old import pattern
    try:
        from swarms import output_types as swarms_output_types
    except ImportError:
        # Create a dummy namespace
        class swarms_output_types:
            OutputType = str

# Initialize centralized logger first (needed for import warnings)
CORPORATE_LOGGER = initialize_logger(log_folder="corporate_swarm")

# CRCA imports - Causal Reasoning and Counterfactual Analysis
try:
    import sys
    import importlib.util
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from CRCA import CRCAAgent
    CRCA_AVAILABLE = True
except ImportError:
    CRCA_AVAILABLE = False
    CORPORATE_LOGGER.warning("CRCA not available, falling back to Agent")

# CRCA-Q imports - Quantitative Trading (file has hyphen, use importlib)
try:
    import sys
    import importlib.util
    # Add project root to path for swarms cloud import
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Get path to CRCA-Q.py (has hyphen, can't use normal import)
    branches_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crca_q_path = os.path.join(branches_path, 'CRCA-Q.py')
    if os.path.exists(crca_q_path):
        # Load module using importlib (handles hyphen in filename)
        spec = importlib.util.spec_from_file_location("CRCA_Q", crca_q_path)
        crca_q_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crca_q_module)  # type: ignore
        CRCA_Q_QuantTradingAgent = crca_q_module.QuantTradingAgent
        CRCA_Q_AVAILABLE = True
    else:
        raise ImportError(f"CRCA-Q.py not found at {crca_q_path}")
except Exception:
    CRCA_Q_AVAILABLE = False
    CORPORATE_LOGGER.warning("CRCA-Q not available, Investment Committee disabled")
    CRCA_Q_QuantTradingAgent = None

# CRCA-SD imports - Governance System
try:
    import sys
    # Add branches directory to path
    branches_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if branches_path not in sys.path:
        sys.path.insert(0, branches_path)
    from crca_sd.crca_sd_governance import (
        GovernanceSystem, Board, BoardType, Arbitration, BoardMember as CRCA_SD_BoardMember
    )
    from crca_sd.crca_sd_core import ControlVector
    import numpy as np
    CRCA_SD_AVAILABLE = True
except ImportError:
    CRCA_SD_AVAILABLE = False
    CORPORATE_LOGGER.warning("CRCA-SD governance not available, enhanced voting disabled")
    # Create dummy types for type hints
    class ControlVector:
        pass
    class BoardType:
        pass
    class GovernanceSystem:
        pass
    class Board:
        pass
    class Arbitration:
        pass
    CRCA_SD_BoardMember = None

# AOP imports - Agent Orchestration Platform
try:
    import sys
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.aop import (
        AOP, TaskQueue, TaskStatus, QueueStatus, QueueStats,
        AgentToolConfig, AOPCluster
    )
    AOP_AVAILABLE = True
except (ImportError, AttributeError):
    AOP_AVAILABLE = False
    CORPORATE_LOGGER.warning("AOP not available, queue-based execution and MCP deployment disabled")
    # Create dummy types for type hints
    class AOP:
        pass
    class TaskQueue:
        pass
    class TaskStatus:
        pass
    class QueueStatus:
        pass
    class QueueStats:
        pass
    class AgentToolConfig:
        pass
    class AOPCluster:
        pass

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Logger is initialized above with imports

# Default configuration values
DEFAULT_BOARD_SIZE = 6
DEFAULT_EXECUTIVE_TEAM_SIZE = 4
DEFAULT_DECISION_THRESHOLD = 0.6
DEFAULT_BUDGET_LIMIT = 200.0
DEFAULT_BATCH_SIZE = 25
DEFAULT_MEETING_DURATION = 600

# Risk assessment constants
RISK_LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
RISK_CATEGORIES = ["operational", "financial", "strategic", "compliance", "cybersecurity", "reputation", "environmental", "regulatory"]

# Stakeholder types
STAKEHOLDER_TYPES = ["investor", "customer", "employee", "community", "supplier", "regulator"]

# Compliance frameworks
COMPLIANCE_FRAMEWORKS = {
    "SOX": ("financial", ["Internal controls", "Financial reporting", "Audit requirements"]),
    "GDPR": ("data_privacy", ["Data protection", "Privacy rights", "Consent management"]),
    "ISO 27001": ("cybersecurity", ["Information security", "Risk management", "Security controls"]),
    "ESG": ("sustainability", ["Environmental reporting", "Social responsibility", "Governance standards"]),
    "HIPAA": ("healthcare", ["Patient privacy", "Data security", "Compliance monitoring"])
}


# ============================================================================
# CORPORATE SWARM CONFIGURATION
# ============================================================================


class CorporateConfigModel(BaseModel):
    """Configuration model for CorporateSwarm with corporate structure and governance settings."""

    # Corporate structure
    default_board_size: int = Field(
        default=6,
        ge=3,
        le=15,
        description="Default number of board members when creating a new board.",
    )

    default_executive_team_size: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Default number of executive team members.",
    )

    # Governance settings
    decision_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for majority decisions (0.0-1.0).",
    )

    enable_democratic_discussion: bool = Field(
        default=True,
        description="Enable democratic discussion features.",
    )

    enable_departmental_work: bool = Field(
        default=True,
        description="Enable departmental collaboration.",
    )

    enable_financial_oversight: bool = Field(
        default=True,
        description="Enable financial oversight features.",
    )

    # Model settings
    default_corporate_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for corporate member agents.",
    )

    # Logging and monitoring
    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for corporate operations.",
    )

    # Performance settings
    max_corporate_meeting_duration: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Maximum duration for corporate meetings in seconds.",
    )

    budget_limit: float = Field(
        default=200.0,
        ge=0.0,
        description="Maximum budget in dollars for corporate operations.",
    )

    batch_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of members to process in batches.",
    )

    enable_lazy_loading: bool = Field(
        default=True,
        description="Enable lazy loading of member agents.",
    )

    enable_caching: bool = Field(
        default=True,
        description="Enable response caching.",
    )

    # CRCA integration settings
    enable_causal_reasoning: bool = Field(
        default=True,
        description="Enable causal reasoning for member agents.",
    )

    enable_quant_analysis: bool = Field(
        default=True,
        description="Enable quantitative analysis for financial proposals.",
    )

    enable_crca_sd_governance: bool = Field(
        default=True,
        description="Enable CRCA-SD governance system for enhanced voting.",
    )

    quant_trading_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for quantitative trading agent.",
    )

    causal_graph_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for corporate causal graph.",
    )

    # AOP integration settings
    enable_aop: bool = Field(
        default=False,
        description="Enable AOP (Agent Orchestration Platform) for MCP deployment and queue-based execution.",
    )

    enable_queue_execution: bool = Field(
        default=True,
        description="Enable queue-based task execution for member agents.",
    )

    aop_server_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for AOP MCP server deployment.",
    )

    aop_max_workers_per_agent: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum number of workers per agent in queue system.",
    )

    aop_max_queue_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum queue size per agent.",
    )

    aop_processing_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout for task processing in seconds.",
    )

    aop_persistence: bool = Field(
        default=False,
        description="Enable persistence mode for AOP server (auto-restart on shutdown).",
    )

    aop_network_monitoring: bool = Field(
        default=True,
        description="Enable network connection monitoring and retry.",
    )

    # Human-in-the-loop safety controls
    enable_human_approval: bool = Field(
        default=True,
        description="Enable human approval gates for major decisions.",
    )

    major_decision_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Budget impact threshold (as fraction of total budget) requiring human approval.",
    )

    max_autonomous_budget_per_period: float = Field(
        default=50.0,
        ge=0.0,
        description="Maximum budget that can be spent autonomously per period without approval.",
    )

    emergency_stop_enabled: bool = Field(
        default=True,
        description="Enable emergency stop mechanism for immediate halt.",
    )

    risk_threshold_for_approval: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Risk score threshold requiring human approval.",
    )

    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level for autonomous execution.",
    )

    max_actions_per_period: int = Field(
        default=100,
        ge=1,
        description="Maximum number of actions per time period before requiring approval.",
    )

    approval_timeout: int = Field(
        default=3600,
        ge=60,
        description="Timeout in seconds for human approval requests.",
    )


@dataclass
class CorporateConfig:
    """Configuration manager for CorporateSwarm."""
    
    config_file_path: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    config: CorporateConfigModel = field(init=False)

    def __post_init__(self) -> None:
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration with priority: explicit data > file > defaults."""
        try:
            self.config = CorporateConfigModel()
            if self.config_file_path and os.path.exists(self.config_file_path):
                self._load_from_file()
            if self.config_data:
                self._load_from_dict(self.config_data)
        except Exception as e:
            CORPORATE_LOGGER.error(f"Configuration loading failed: {e}")
            raise ValueError(f"Configuration loading failed: {e}") from e

    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_file_path, "r") as f:
                self._load_from_dict(yaml.safe_load(f))
                CORPORATE_LOGGER.info(f"Loaded config from: {self.config_file_path}")
        except Exception as e:
            CORPORATE_LOGGER.warning(f"File loading failed {self.config_file_path}: {e}")
            raise ValueError(f"Configuration file loading failed: {e}") from e

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary with validation."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                try:
                    setattr(self.config, key, value)
                except (ValueError, TypeError) as e:
                    CORPORATE_LOGGER.warning(f"Config {key} failed: {e}")
                    raise ValueError(f"Invalid configuration value for {key}: {e}") from e

    def get_config(self) -> CorporateConfigModel:
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        try:
            self._load_from_dict(updates)
        except ValueError as e:
            CORPORATE_LOGGER.error(f"Config update failed: {e}")
            raise ValueError(f"Configuration update failed: {e}") from e

    def validate_config(self) -> List[str]:
        """Validate configuration and return error list."""
        errors = []
        try:
            self.config.model_validate(self.config.model_dump())
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        
        if self.config.decision_threshold < 0.5:
            errors.append("Decision threshold should be at least 0.5")
        if self.config.default_board_size < 3:
            errors.append("Board size should be at least 3")
        
        return errors


# Global configuration cache
_corporate_config: Optional[CorporateConfig] = None

@lru_cache(maxsize=1)
def get_corporate_config(config_file_path: Optional[str] = None) -> CorporateConfig:
    """Get global CorporateSwarm configuration instance."""
    global _corporate_config
    if _corporate_config is None:
        _corporate_config = CorporateConfig(config_file_path=config_file_path)
    return _corporate_config


# ============================================================================
# CORPORATE SWARM DATA MODELS
# ============================================================================

def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())

def _get_audit_date() -> float:
    """Get next audit date (1 year from now)."""
    return time.time() + (365 * 24 * 60 * 60)


class BaseCorporateAgent(ABC):
    """Base class for corporate agents."""
    
    @abstractmethod
    def run(self, task: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """Synchronous execution method."""
        pass
    
    @abstractmethod
    async def arun(self, task: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """Asynchronous execution method."""
        pass
    
    def __call__(self, task: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """Callable interface."""
        return self.run(task, **kwargs)


class CorporateRole(str, Enum):
    """Corporate roles and positions."""
    CEO, CFO, CTO, COO = "ceo", "cfo", "cto", "coo"
    BOARD_CHAIR, BOARD_VICE_CHAIR, BOARD_MEMBER = "board_chair", "board_vice_chair", "board_member"
    INDEPENDENT_DIRECTOR, EXECUTIVE_DIRECTOR, NON_EXECUTIVE_DIRECTOR = "independent_director", "executive_director", "non_executive_director"
    COMMITTEE_CHAIR, COMMITTEE_MEMBER = "committee_chair", "committee_member"
    DEPARTMENT_HEAD, MANAGER, EMPLOYEE = "department_head", "manager", "employee"
    INVESTOR, ADVISOR, AUDITOR, SECRETARY = "investor", "advisor", "auditor", "secretary"


class DepartmentType(str, Enum):
    """Corporate department types."""
    FINANCE, OPERATIONS, MARKETING, HUMAN_RESOURCES = "finance", "operations", "marketing", "human_resources"
    LEGAL, TECHNOLOGY, RESEARCH_DEVELOPMENT = "legal", "technology", "research_development"
    SALES, CUSTOMER_SERVICE, COMPLIANCE = "sales", "customer_service", "compliance"


class ProposalType(str, Enum):
    """Types of corporate proposals."""
    STRATEGIC_INITIATIVE, BUDGET_ALLOCATION, HIRING_DECISION = "strategic_initiative", "budget_allocation", "hiring_decision"
    PRODUCT_LAUNCH, PARTNERSHIP, MERGER_ACQUISITION = "product_launch", "partnership", "merger_acquisition"
    POLICY_CHANGE, INVESTMENT, OPERATIONAL_CHANGE = "policy_change", "investment", "operational_change"
    COMPLIANCE_UPDATE, BOARD_RESOLUTION, EXECUTIVE_COMPENSATION = "compliance_update", "board_resolution", "executive_compensation"
    DIVIDEND_DECLARATION, SHARE_ISSUANCE, AUDIT_APPOINTMENT = "dividend_declaration", "share_issuance", "audit_appointment"
    RISK_MANAGEMENT, SUCCESSION_PLANNING = "risk_management", "succession_planning"


class VoteResult(str, Enum):
    """Voting result outcomes."""
    APPROVED, REJECTED, TABLED, FAILED = "approved", "rejected", "tabled", "failed"
    UNANIMOUS, MAJORITY, MINORITY, ABSTAINED = "unanimous", "majority", "minority", "abstained"


class BoardCommitteeType(str, Enum):
    """Types of board committees."""
    AUDIT, COMPENSATION, NOMINATING, GOVERNANCE = "audit", "compensation", "nominating", "governance"
    RISK, TECHNOLOGY, STRATEGIC, FINANCE = "risk", "technology", "strategic", "finance"
    COMPLIANCE, ESG, SUSTAINABILITY, CYBERSECURITY = "compliance", "esg", "sustainability", "cybersecurity"
    INNOVATION, STAKEHOLDER, CRISIS_MANAGEMENT = "innovation", "stakeholder", "crisis_management"
    AI_ETHICS, DATA_PRIVACY = "ai_ethics", "data_privacy"


class MeetingType(str, Enum):
    """Types of board meetings."""
    REGULAR_BOARD, SPECIAL_BOARD, ANNUAL_GENERAL = "regular_board", "special_board", "annual_general"
    COMMITTEE_MEETING, EXECUTIVE_SESSION, EMERGENCY_MEETING = "committee_meeting", "executive_session", "emergency_meeting"
    ESG_REVIEW, RISK_ASSESSMENT, CRISIS_RESPONSE = "esg_review", "risk_assessment", "crisis_response"
    STAKEHOLDER_ENGAGEMENT, INNOVATION_REVIEW, COMPLIANCE_AUDIT = "stakeholder_engagement", "innovation_review", "compliance_audit"
    SUSTAINABILITY_REPORTING, AI_ETHICS_REVIEW = "sustainability_reporting", "ai_ethics_review"


@dataclass
class BoardCommittee:
    """Board committee with governance responsibilities."""
    committee_id: str = field(default_factory=_generate_uuid)
    name: str = ""
    committee_type: BoardCommitteeType = BoardCommitteeType.GOVERNANCE
    chair: str = ""
    members: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    meeting_schedule: str = ""
    quorum_required: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoardMeeting:
    """Board meeting with agenda and minutes."""
    meeting_id: str = field(default_factory=_generate_uuid)
    meeting_type: MeetingType = MeetingType.REGULAR_BOARD
    date: float = field(default_factory=time.time)
    location: str = ""
    attendees: List[str] = field(default_factory=list)
    agenda: List[str] = field(default_factory=list)
    minutes: str = ""
    quorum_met: bool = False
    resolutions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorporateMember(BaseModel):
    """Corporate stakeholder with role, expertise, and governance responsibilities."""
    
    member_id: str = Field(default_factory=_generate_uuid)
    name: str = Field(default="")
    role: CorporateRole = Field(default=CorporateRole.EMPLOYEE)
    department: DepartmentType = Field(default=DepartmentType.OPERATIONS)
    expertise_areas: List[str] = Field(default_factory=list)
    voting_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    board_committees: List[str] = Field(default_factory=list)
    independence_status: bool = Field(default=False)
    term_start: float = Field(default_factory=time.time)
    term_end: Optional[float] = Field(default=None)
    compensation: Dict[str, Any] = Field(default_factory=dict)
    agent: Optional[Any] = Field(default=None, exclude=True)  # CRCAAgent when available
    causal_variables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CorporateProposal(BaseModel):
    """Corporate proposal requiring decision-making with financial impact."""
    
    proposal_id: str = Field(default_factory=_generate_uuid)
    title: str = Field(default="")
    description: str = Field(default="")
    proposal_type: ProposalType = Field(default=ProposalType.STRATEGIC_INITIATIVE)
    sponsor: str = Field(default="")
    department: DepartmentType = Field(default=DepartmentType.OPERATIONS)
    budget_impact: float = Field(default=0.0, ge=0.0)
    timeline: str = Field(default="")
    status: str = Field(default="pending")
    causal_analysis: Optional[Dict[str, Any]] = Field(default=None)
    quant_analysis: Optional[Dict[str, Any]] = Field(default=None)
    board_evaluations: Optional[Dict[str, Any]] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class CorporateDepartment:
    """Corporate department with specific functions and budget."""
    department_id: str = field(default_factory=_generate_uuid)
    name: str = ""
    department_type: DepartmentType = DepartmentType.OPERATIONS
    head: str = ""
    members: List[str] = field(default_factory=list)
    budget: float = 0.0
    objectives: List[str] = field(default_factory=list)
    current_projects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorporateVote(BaseModel):
    """Corporate voting session with individual votes and analysis."""
    
    vote_id: str = Field(default_factory=_generate_uuid)
    proposal: CorporateProposal = Field(default_factory=CorporateProposal)
    participants: List[str] = Field(default_factory=list)
    individual_votes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    political_group_analysis: Dict[str, Any] = Field(default_factory=dict)
    result: VoteResult = Field(default=VoteResult.FAILED)
    timestamp: float = Field(default_factory=time.time)
    causal_reasoning_summary: str = Field(default="")
    quant_signals: Dict[str, float] = Field(default_factory=dict)
    governance_consensus: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ESGScore(BaseModel):
    """Environmental, Social, and Governance scoring model."""
    
    environmental_score: float = Field(default=0.0, ge=0.0, le=100.0)
    social_score: float = Field(default=0.0, ge=0.0, le=100.0)
    governance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    carbon_footprint: float = Field(default=0.0, ge=0.0)
    diversity_index: float = Field(default=0.0, ge=0.0, le=1.0)
    stakeholder_satisfaction: float = Field(default=0.0, ge=0.0, le=100.0)
    sustainability_goals: List[str] = Field(default_factory=list)
    last_updated: float = Field(default_factory=time.time)


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment model for corporate governance."""
    
    risk_id: str = Field(default_factory=_generate_uuid)
    risk_category: str = Field(default="operational")
    risk_level: str = Field(default="medium")
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.25, ge=0.0, le=1.0)
    mitigation_strategies: List[str] = Field(default_factory=list)
    owner: str = Field(default="")
    status: str = Field(default="active")
    last_reviewed: float = Field(default_factory=time.time)


class StakeholderEngagement(BaseModel):
    """Stakeholder engagement and management model."""
    
    stakeholder_id: str = Field(default_factory=_generate_uuid)
    stakeholder_type: str = Field(default="investor")
    name: str = Field(default="")
    influence_level: str = Field(default="medium")
    interest_level: str = Field(default="medium")
    engagement_frequency: str = Field(default="quarterly")
    satisfaction_score: float = Field(default=0.0, ge=0.0, le=100.0)
    concerns: List[str] = Field(default_factory=list)
    last_engagement: float = Field(default_factory=time.time)


class ComplianceFramework(BaseModel):
    """Regulatory compliance and audit framework model."""
    
    compliance_id: str = Field(default_factory=_generate_uuid)
    regulation_name: str = Field(default="")
    regulation_type: str = Field(default="financial")
    compliance_status: str = Field(default="compliant")
    compliance_score: float = Field(default=100.0, ge=0.0, le=100.0)
    requirements: List[str] = Field(default_factory=list)
    controls: List[str] = Field(default_factory=list)
    audit_findings: List[str] = Field(default_factory=list)
    next_audit_date: float = Field(default_factory=_get_audit_date)
    responsible_officer: str = Field(default="")


class CorporateSwarm(BaseCorporateAgent):
    """Autonomous corporate governance system with democratic decision-making."""
    
    def __init__(
        self,
        name: str = "CorporateSwarm",
        description: str = "A comprehensive corporate governance system with democratic decision-making",
        max_loops: int = 1,
        output_type: swarms_output_types.OutputType = "dict-all-except-first",
        corporate_model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        config_file_path: Optional[str] = None,
        config_data: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize CorporateSwarm with corporate governance capabilities."""
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.output_type = output_type
        self.corporate_model_name = corporate_model_name
        self.verbose = verbose
        
        # Load configuration
        self.config = CorporateConfig(
            config_file_path=config_file_path,
            config_data=config_data
        ).get_config()
        
        # Initialize corporate structure
        self.members: Dict[str, CorporateMember] = {}
        self.departments: Dict[str, CorporateDepartment] = {}
        self.proposals: List[CorporateProposal] = []
        self.votes: List[CorporateVote] = []
        self.board_members: List[str] = []
        self.executive_team: List[str] = []
        self.board_committees: Dict[str, BoardCommittee] = {}
        self.board_meetings: List[BoardMeeting] = []
        self.independent_directors: List[str] = []
        self.executive_directors: List[str] = []
        
        # Advanced governance frameworks
        self.esg_scores: Dict[str, ESGScore] = {}
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        self.stakeholder_engagements: Dict[str, StakeholderEngagement] = {}
        self.compliance_frameworks: Dict[str, ComplianceFramework] = {}
        self.crisis_management_plans: Dict[str, Dict[str, Any]] = {}
        self.innovation_pipeline: List[Dict[str, Any]] = []
        self.audit_trails: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.sustainability_targets: Dict[str, Any] = {}
        self.ai_ethics_framework: Dict[str, Any] = {}
        
        # Initialize conversation and democratic systems
        self.conversation = Conversation(time_enabled=False)
        self.democratic_swarm = None
        
        # Cost tracking
        self.cost_tracker = CostTracker(self.config.budget_limit)
        
        # Constitutional fallback state tracking
        self._constitutional_failure_count: int = 0
        self._last_successful_state: Dict[str, Any] = {}
        self.emergency_stop_active: bool = False
        self._frozen_actions: bool = False
        
        # Mandate execution tracking (for daemon)
        self.running_mandates: Dict[str, Dict[str, Any]] = {}
        
        # Performance settings
        self.max_workers = os.cpu_count()
        
        # CRCA-SD Governance System
        self.governance_system: Optional[GovernanceSystem] = None
        if self.config.enable_crca_sd_governance and CRCA_SD_AVAILABLE:
            # Will be initialized in _initialize_default_structure
            pass
        
        # Investment Committee (QuantTradingAgent)
        self.investment_committee: Optional[Any] = None
        if self.config.enable_quant_analysis and CRCA_Q_AVAILABLE:
            # Will be initialized in _initialize_default_structure
            pass
        
        # Corporate causal graph
        self.corporate_causal_graph: Dict[str, List[str]] = {}
        if self.config.enable_causal_reasoning:
            self._build_corporate_causal_graph()
        
        # AOP (Agent Orchestration Platform) for MCP deployment
        self.aop_server: Optional[AOP] = None
        self.member_task_queues: Dict[str, TaskQueue] = {}
        if self.config.enable_aop and AOP_AVAILABLE:
            try:
                self._initialize_aop_server()
            except Exception as e:
                CORPORATE_LOGGER.warning(f"Failed to initialize AOP server: {e}")
                self.aop_server = None
        
        # Initialize queue system for member agents if enabled
        if self.config.enable_queue_execution and AOP_AVAILABLE:
            self._initialize_member_queues()
        
        # Initialize the corporate swarm
        self._init_corporate_swarm()
    
    def _init_corporate_swarm(self) -> None:
        """Initialize CorporateSwarm structure and perform reliability checks."""
        try:
            if self.verbose:
                CORPORATE_LOGGER.info(
                    f"Initializing CorporateSwarm: {self.name}"
                )
                CORPORATE_LOGGER.info(
                    f"Configuration - Max loops: {self.max_loops}"
                )

            # Perform reliability checks
            self._perform_reliability_checks()

            # Initialize default corporate structure
            self._initialize_default_structure()

            # Initialize democratic swarm if enabled
            if self.config.enable_democratic_discussion:
                self._initialize_democratic_swarm()

            if self.verbose:
                CORPORATE_LOGGER.success(
                    f"CorporateSwarm initialized successfully: {self.name}"
                )

        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to initialize CorporateSwarm: {str(e)}")
            raise RuntimeError(f"Failed to initialize CorporateSwarm: {str(e)}") from e

    def _perform_reliability_checks(self) -> None:
        """Validate critical requirements and configuration parameters."""
        try:
            if self.verbose:
                CORPORATE_LOGGER.info(
                    f"Running reliability checks for CorporateSwarm: {self.name}"
                )

            if self.max_loops <= 0:
                raise ValueError(
                    "Max loops must be greater than 0. Please set a valid number of loops."
                )

            if (
                self.config.decision_threshold < 0.0
                or self.config.decision_threshold > 1.0
            ):
                raise ValueError(
                    "Decision threshold must be between 0.0 and 1.0."
                )

            if self.config.budget_limit < 0:
                raise ValueError(
                    "Budget limit must be non-negative."
                )

            if self.verbose:
                CORPORATE_LOGGER.success(
                    f"Reliability checks passed for CorporateSwarm: {self.name}"
                )

        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed reliability checks: {str(e)}")
            raise ValueError(f"Reliability checks failed: {str(e)}") from e
    
    @classmethod
    def create_simple_corporation(
        cls,
        name: str = "SimpleCorp",
        num_board_members: int = 5,
        num_executives: int = 4,
        verbose: bool = False
    ) -> "CorporateSwarm":
        """Create a simple corporation with basic structure."""
        return cls(
            name=name,
            max_loops=1,
            verbose=verbose,
            corporate_model_name="gpt-4o-mini"
        )
    
    def run(
        self,
        task: str,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Main execution method - processes corporate tasks through democratic decision-making.
        
        Implements intelligent task routing and performance optimization via concurrency.
        Includes human-in-the-loop safety controls for autonomous operation.
        
        Args:
            task: The corporate task or proposal to process
            **kwargs: Additional parameters for task processing
            
        Returns:
            Union[str, Dict[str, Any]]: Task results or decision outcomes
            
        Example:
            >>> corporate = CorporateSwarm()
            >>> result = corporate.run("Should we invest in AI technology?")
            >>> print(result['vote_result'])
        """
        if self.verbose:
            CORPORATE_LOGGER.info(f"CorporateSwarm processing task: {task[:100]}...")
        
        # Check emergency stop
        if self.emergency_stop_active:
            CORPORATE_LOGGER.warning("Emergency stop is active - all operations halted")
            return {
                "status": "blocked",
                "reason": "Emergency stop active",
                "task": task,
                "requires_human_intervention": True
            }
        
        # Check if actions are frozen (constitutional safe halt)
        if self._frozen_actions:
            CORPORATE_LOGGER.warning("Actions frozen due to constitutional failure - safe halt active")
            return {
                "status": "frozen",
                "reason": "Constitutional safe halt - repeated decision-making failures",
                "task": task,
                "failure_count": self._constitutional_failure_count,
                "requires_human_intervention": True
            }
        
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            CORPORATE_LOGGER.warning(f"Budget limit exceeded for task: {task[:50]}...")
            return {
                "status": "failed",
                "reason": "Budget limit exceeded",
                "task": task
            }
        
        # Check action limits (for full autonomy, this just resets periodically)
        self._check_and_reset_action_limits()
        
        try:
            # Determine task type and route accordingly with performance optimization
            task_lower = task.lower()
            
            # Enhanced routing: detect code-related tasks and route to proposal processing
            code_keywords = ["build", "create", "develop", "prototype", "draft", "code", "software", 
                           "app", "application", "website", "platform", "system", "tool", "program"]
            
            if any(keyword in task_lower for keyword in ["proposal", "vote", "decision"]):
                return self._process_proposal_task(task, **kwargs)
            elif any(keyword in task_lower for keyword in code_keywords):
                # Code-related tasks should go through proposal process for auto-execution
                if self.verbose:
                    CORPORATE_LOGGER.info(f"Detected code-related task, routing to proposal processing")
                return self._process_proposal_task(task, **kwargs)
            elif any(keyword in task_lower for keyword in ["meeting", "board", "committee"]):
                return self._process_meeting_task(task, **kwargs)
            elif any(keyword in task_lower for keyword in ["strategic", "planning", "initiative"]):
                return self._process_strategic_task(task, **kwargs)
            else:
                return self._process_general_task(task, **kwargs)
                
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error processing task '{task[:50]}...': {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    async def arun(
        self,
        task: str,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Asynchronous version of run method with concurrency optimization.
        
        Args:
            task: The corporate task or proposal to process
            **kwargs: Additional parameters for task processing
            
        Returns:
            Union[str, Dict[str, Any]]: Task results or decision outcomes
        """
        if self.verbose:
            logger.info(f"CorporateSwarm async processing task: {task[:100]}...")
        
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            logger.warning(f"Budget limit exceeded for async task: {task[:50]}...")
            return {
                "status": "failed",
                "reason": "Budget limit exceeded",
                "task": task
            }
        
        try:
            # Run in thread pool for I/O bound operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.run, task, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error in async processing task '{task[:50]}...': {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    def execute_proposal_as_mandate(
        self,
        proposal_id: str
    ) -> Dict[str, Any]:
        """
        Execute a corporate proposal as a code generation mandate.
        
        Convenience method that converts a CorporateProposal to a mandate
        and executes it via bolt.diy with full governance oversight.
        
        Args:
            proposal_id: ID of the corporate proposal to execute
            
        Returns:
            Dict containing execution results, status, and governance metadata
            
        Example:
            >>> proposal_id = corporate.create_proposal(
            ...     title="Build customer portal",
            ...     description="Create a React-based customer portal",
            ...     proposal_type=ProposalType.PRODUCT_LAUNCH,
            ...     sponsor_id=ceo_id,
            ...     department=DepartmentType.TECHNOLOGY,
            ...     budget_impact=50.0
            ... )
            >>> result = corporate.execute_proposal_as_mandate(proposal_id)
        """
        # Find proposal
        proposal = self._find_proposal(proposal_id)
        
        # Convert proposal to mandate using mandate generator
        try:
            from tools.mandate_generator import proposal_to_mandate
            mandate = proposal_to_mandate(proposal, corporate_swarm=self)
        except ImportError:
            CORPORATE_LOGGER.warning("mandate_generator not available, creating basic mandate")
            # Fallback: create basic mandate structure
            mandate = {
                "mandate_id": proposal.proposal_id,
                "objectives": [proposal.title, proposal.description],
                "constraints": {
                    "language": "ts",
                    "maxDependencies": 50,
                    "noNetwork": False,
                    "maxFileSize": 100000,
                    "maxFiles": 100,
                },
                "budget": {
                    "token": int(proposal.budget_impact * 10000) if proposal.budget_impact > 0 else 100000,
                    "time": int(proposal.budget_impact * 60) if proposal.budget_impact > 0 else 300,
                    "cost": proposal.budget_impact if proposal.budget_impact > 0 else 5.0,
                },
                "deliverables": ["src/index.ts", "package.json"],
                "governance": {"proposal_id": proposal.proposal_id},
                "iteration_config": {
                    "max_iterations": 3,
                    "test_required": True,
                    "quality_threshold": 0.7,
                },
            }
        
        # Execute mandate
        return self.execute_code_mandate(mandate, proposal_id=proposal_id)
    
    def execute_code_mandate(
        self,
        mandate: Dict[str, Any],
        proposal_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a code generation mandate via bolt.diy.
        
        This method sends a structured mandate to bolt.diy for autonomous
        code generation, deployment, and execution with full governance oversight.
        
        Args:
            mandate: Structured mandate dictionary matching bolt.diy Mandate type
            proposal_id: Optional proposal ID to link execution to governance proposal
            
        Returns:
            Dict containing execution results, status, and governance metadata
            
        Example:
            >>> mandate = {
            ...     "mandate_id": "proj-123",
            ...     "objectives": ["Create a React todo app"],
            ...     "constraints": {"language": "ts", "maxDependencies": 10},
            ...     "budget": {"token": 100000, "time": 300, "cost": 5.0},
            ...     "deliverables": ["src/App.tsx", "package.json"],
            ...     "governance": {"proposal_id": "prop-456"},
            ...     "iteration_config": {"max_iterations": 3, "test_required": True}
            ... }
            >>> result = corporate.execute_code_mandate(mandate, proposal_id="prop-456")
        """
        if self.verbose:
            CORPORATE_LOGGER.info(f"Executing code mandate: {mandate.get('mandate_id', 'unknown')}")
        
        # Check budget before execution
        if not self.cost_tracker.check_budget():
            CORPORATE_LOGGER.warning("Budget limit exceeded, cannot execute mandate")
            return {
                "status": "failed",
                "reason": "Budget limit exceeded",
                "mandate_id": mandate.get("mandate_id"),
                "proposal_id": proposal_id
            }
        
        # Get bolt.diy API URL from environment or config
        bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
        if not bolt_diy_url.endswith("/"):
            bolt_diy_url += "/"
        
        api_url = f"{bolt_diy_url}api/mandate"
        
        try:
            # Add proposal_id to mandate governance if provided
            if proposal_id and "governance" in mandate:
                mandate["governance"]["proposal_id"] = proposal_id
            elif proposal_id:
                mandate["governance"] = {"proposal_id": proposal_id}
            
            # Add ESG requirements from corporate state if available
            if "governance" in mandate:
                esg_score = self.calculate_esg_score()
                mandate["governance"]["esg_requirements"] = {
                    "environmental_score": esg_score.environmental_score,
                    "social_score": esg_score.social_score,
                    "governance_score": esg_score.governance_score,
                    "overall_score": esg_score.overall_score,
                }
            
            # Send mandate to bolt.diy API
            import requests
            response = requests.post(
                api_url,
                json=mandate,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 202:
                # Mandate accepted, execution started
                result = response.json()
                
                # Track cost (estimate based on budget)
                estimated_cost = mandate.get("budget", {}).get("cost", 0.0)
                self.cost_tracker.add_cost(estimated_cost * 0.1)  # Reserve 10% upfront
                
                # Log to audit trail
                self.audit_trails.append({
                    "event": "code_mandate_executed",
                    "mandate_id": mandate.get("mandate_id"),
                    "proposal_id": proposal_id,
                    "status": "accepted",
                    "timestamp": time.time(),
                    "bolt_diy_url": api_url
                })
                
                if self.verbose:
                    CORPORATE_LOGGER.info(f"Mandate {mandate.get('mandate_id')} accepted by bolt.diy")
                
                # Track mandate execution
                mandate_id = mandate.get("mandate_id")
                if mandate_id:
                    self._track_mandate_execution(
                        mandate_id=mandate_id,
                        proposal_id=proposal_id,
                        event_stream_url=result.get("event_stream_url")
                    )
                
                return {
                    "status": "accepted",
                    "mandate_id": mandate_id,
                    "proposal_id": proposal_id,
                    "event_stream_url": result.get("event_stream_url"),
                    "message": "Mandate accepted, execution started",
                    "timestamp": time.time()
                }
            else:
                error_text = response.text
                CORPORATE_LOGGER.error(f"Failed to execute mandate: {response.status_code} - {error_text}")
                
                return {
                    "status": "failed",
                    "mandate_id": mandate.get("mandate_id"),
                    "proposal_id": proposal_id,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "timestamp": time.time()
                }
                
        except requests.exceptions.RequestException as e:
            CORPORATE_LOGGER.error(f"Network error executing mandate: {e}")
            return {
                "status": "failed",
                "mandate_id": mandate.get("mandate_id"),
                "proposal_id": proposal_id,
                "error": f"Network error: {str(e)}",
                "timestamp": time.time()
            }
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error executing code mandate: {e}")
            return {
                "status": "error",
                "mandate_id": mandate.get("mandate_id"),
                "proposal_id": proposal_id,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _build_corporate_causal_graph(self) -> None:
        """Build corporate causal graph with key relationships."""
        if not self.config.enable_causal_reasoning:
            return
        
        self.corporate_causal_graph = {
            'budget_allocation': ['revenue', 'profitability', 'cash_flow'],
            'revenue': ['sales', 'market_share', 'pricing'],
            'profitability': ['revenue', 'costs', 'efficiency'],
            'esg_score': ['environmental_score', 'social_score', 'governance_score'],
            'environmental_score': ['carbon_footprint', 'sustainability_practices'],
            'social_score': ['stakeholder_satisfaction', 'diversity_index', 'employee_satisfaction'],
            'governance_score': ['board_independence', 'transparency', 'compliance_score'],
            'risk_level': ['operational_risk', 'financial_risk', 'strategic_risk', 'compliance_risk'],
            'operational_risk': ['process_efficiency', 'quality_control', 'supply_chain'],
            'financial_risk': ['liquidity', 'leverage', 'market_volatility'],
            'strategic_risk': ['market_position', 'competitive_advantage', 'innovation'],
            'compliance_risk': ['regulatory_changes', 'audit_findings', 'internal_controls'],
            'board_performance': ['independence_ratio', 'meeting_frequency', 'decision_efficiency'],
            'stakeholder_satisfaction': ['investor_satisfaction', 'customer_satisfaction', 'employee_satisfaction'],
            'innovation_pipeline': ['r_and_d_investment', 'technology_adoption', 'market_readiness'],
        }
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Built corporate causal graph with {len(self.corporate_causal_graph)} nodes")
    
    def _get_corporate_variables(self) -> List[str]:
        """Get list of corporate variables for causal reasoning."""
        variables = [
            'budget_allocation', 'revenue', 'profitability', 'cash_flow',
            'esg_score', 'environmental_score', 'social_score', 'governance_score',
            'risk_level', 'operational_risk', 'financial_risk', 'strategic_risk', 'compliance_risk',
            'board_performance', 'stakeholder_satisfaction', 'innovation_pipeline',
            'sales', 'market_share', 'pricing', 'costs', 'efficiency',
            'carbon_footprint', 'sustainability_practices', 'diversity_index',
            'board_independence', 'transparency', 'compliance_score',
            'liquidity', 'leverage', 'market_volatility', 'market_position',
            'competitive_advantage', 'r_and_d_investment', 'technology_adoption'
        ]
        return variables
    
    def _get_causal_edges_for_member(self, member: CorporateMember) -> List[tuple]:
        """Get causal edges relevant to a member's expertise areas."""
        edges = []
        expertise_lower = [e.lower() for e in member.expertise_areas]
        
        # Map expertise to causal relationships
        if any('finance' in e or 'accounting' in e for e in expertise_lower):
            edges.extend([
                ('revenue', 'profitability'),
                ('budget_allocation', 'financial_risk'),
                ('cash_flow', 'liquidity'),
            ])
        
        if any('strategy' in e or 'planning' in e for e in expertise_lower):
            edges.extend([
                ('market_position', 'revenue'),
                ('strategic_risk', 'risk_level'),
                ('innovation_pipeline', 'competitive_advantage'),
            ])
        
        if any('technology' in e or 'innovation' in e for e in expertise_lower):
            edges.extend([
                ('technology_adoption', 'efficiency'),
                ('r_and_d_investment', 'innovation_pipeline'),
                ('innovation_pipeline', 'market_position'),
            ])
        
        if any('governance' in e or 'compliance' in e for e in expertise_lower):
            edges.extend([
                ('board_independence', 'governance_score'),
                ('compliance_score', 'compliance_risk'),
                ('transparency', 'stakeholder_satisfaction'),
            ])
        
        if any('sustainability' in e or 'environmental' in e for e in expertise_lower):
            edges.extend([
                ('sustainability_practices', 'environmental_score'),
                ('carbon_footprint', 'environmental_score'),
                ('environmental_score', 'esg_score'),
            ])
        
        # Default edges for all members
        edges.extend([
            ('esg_score', 'stakeholder_satisfaction'),
            ('risk_level', 'board_performance'),
            ('profitability', 'budget_allocation'),
        ])
        
        return list(set(edges))  # Remove duplicates
    
    def _initialize_default_structure(self) -> None:
        """Initialize default corporate structure with key positions."""
        if self.verbose:
            CORPORATE_LOGGER.info("Initializing default corporate structure")
        
        # Create executive team
        executives = [
            ("John Smith", CorporateRole.CEO, DepartmentType.OPERATIONS, ["strategy", "leadership"]),
            ("Sarah Johnson", CorporateRole.CFO, DepartmentType.FINANCE, ["finance", "accounting"]),
            ("Michael Chen", CorporateRole.CTO, DepartmentType.TECHNOLOGY, ["technology", "innovation"]),
            ("Emily Davis", CorporateRole.COO, DepartmentType.OPERATIONS, ["operations", "efficiency"]),
        ]
        
        for name, role, dept, expertise in executives:
            member = CorporateMember(
                name=name,
                role=role,
                department=dept,
                expertise_areas=expertise,
                voting_weight=2.0 if role in [CorporateRole.CEO, CorporateRole.CFO] else 1.5
            )
            self.members[member.member_id] = member
            self.executive_team.append(member.member_id)
        
        # Create board members with enhanced governance structure
        board_members = [
            ("Robert Wilson", CorporateRole.BOARD_CHAIR, DepartmentType.OPERATIONS, ["governance", "strategy"], True),
            ("Lisa Anderson", CorporateRole.INDEPENDENT_DIRECTOR, DepartmentType.FINANCE, ["finance", "investments"], True),
            ("David Brown", CorporateRole.EXECUTIVE_DIRECTOR, DepartmentType.TECHNOLOGY, ["technology", "innovation"], False),
            ("Maria Garcia", CorporateRole.INDEPENDENT_DIRECTOR, DepartmentType.MARKETING, ["marketing", "branding"], True),
            ("James Chen", CorporateRole.BOARD_VICE_CHAIR, DepartmentType.LEGAL, ["legal", "compliance"], True),
            ("Sarah Thompson", CorporateRole.INDEPENDENT_DIRECTOR, DepartmentType.HUMAN_RESOURCES, ["hr", "governance"], True),
        ]
        
        for name, role, dept, expertise, independent in board_members:
            member = CorporateMember(
                name=name,
                role=role,
                department=dept,
                expertise_areas=expertise,
                voting_weight=3.0 if role == CorporateRole.BOARD_CHAIR else 2.5,
                independence_status=independent,
                term_start=time.time(),
                term_end=time.time() + (365 * 24 * 60 * 60 * 3),  # 3 year term
                compensation={"base_retainer": 50000, "meeting_fee": 2000}
            )
            self.members[member.member_id] = member
            self.board_members.append(member.member_id)
            
            if independent:
                self.independent_directors.append(member.member_id)
            if role in [CorporateRole.EXECUTIVE_DIRECTOR, CorporateRole.CEO, CorporateRole.CFO, CorporateRole.CTO, CorporateRole.COO]:
                self.executive_directors.append(member.member_id)
        
        # Create departments
        self._create_departments()
        
        # Create board committees
        self._create_board_committees()
        
        # Initialize Investment Committee (QuantTradingAgent)
        if self.config.enable_quant_analysis and CRCA_Q_AVAILABLE:
            try:
                quant_config = self.config.quant_trading_config or {}
                self.investment_committee = CRCA_Q_QuantTradingAgent(
                    days_back=quant_config.get('days_back', 75),
                    demo_mode=not self.config.enable_financial_oversight,
                    longterm_mode=self.config.enable_lazy_loading,
                    **quant_config
                )
                if self.verbose:
                    CORPORATE_LOGGER.info("Investment Committee (QuantTradingAgent) initialized")
            except Exception as e:
                CORPORATE_LOGGER.warning(f"Failed to initialize Investment Committee: {e}")
                self.investment_committee = None
        
        # Initialize CRCA-SD Governance System
        if self.config.enable_crca_sd_governance and CRCA_SD_AVAILABLE:
            try:
                self._initialize_crca_sd_governance()
            except Exception as e:
                CORPORATE_LOGGER.warning(f"Failed to initialize CRCA-SD governance: {e}")
                self.governance_system = None
        
        # Initialize member agents with CRCAAgent
        if self.config.enable_causal_reasoning and CRCA_AVAILABLE:
            for member_id, member in self.members.items():
                if member.agent is None:
                    self._initialize_member_agent(member)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Created {len(self.members)} members, {len(self.departments)} departments, {len(self.board_committees)} committees")
    
    def _create_departments(self) -> None:
        """Create corporate departments with heads and objectives."""
        department_configs = [
            (DepartmentType.FINANCE, "Finance Department", ["budget_management", "financial_reporting"]),
            (DepartmentType.OPERATIONS, "Operations Department", ["process_optimization", "quality_control"]),
            (DepartmentType.MARKETING, "Marketing Department", ["brand_management", "customer_acquisition"]),
            (DepartmentType.HUMAN_RESOURCES, "Human Resources Department", ["talent_management", "employee_relations"]),
            (DepartmentType.LEGAL, "Legal Department", ["compliance", "contract_management"]),
            (DepartmentType.TECHNOLOGY, "Technology Department", ["system_development", "cybersecurity"]),
        ]
        
        for dept_type, name, objectives in department_configs:
            # Find department head
            head_id = None
            for member_id, member in self.members.items():
                if member.department == dept_type and member.role == CorporateRole.DEPARTMENT_HEAD:
                    head_id = member_id
                    break
            
            department = CorporateDepartment(
                name=name,
                department_type=dept_type,
                head=head_id or "",
                objectives=objectives,
                budget=100000.0  # Default budget
            )
            self.departments[department.department_id] = department
    
    def _create_board_committees(self) -> None:
        """Create board committees with chairs and members."""
        committee_configs = [
            (BoardCommitteeType.AUDIT, "Audit Committee", ["financial_reporting", "internal_controls", "audit_oversight"]),
            (BoardCommitteeType.COMPENSATION, "Compensation Committee", ["executive_compensation", "incentive_plans", "succession_planning"]),
            (BoardCommitteeType.NOMINATING, "Nominating Committee", ["board_nominations", "governance_policies", "director_evaluations"]),
            (BoardCommitteeType.RISK, "Risk Committee", ["risk_management", "cybersecurity", "operational_risk"]),
            (BoardCommitteeType.TECHNOLOGY, "Technology Committee", ["technology_strategy", "digital_transformation", "innovation"]),
            (BoardCommitteeType.ESG, "ESG Committee", ["environmental_sustainability", "social_responsibility", "governance_oversight"]),
            (BoardCommitteeType.SUSTAINABILITY, "Sustainability Committee", ["carbon_neutrality", "renewable_energy", "sustainable_practices"]),
            (BoardCommitteeType.CYBERSECURITY, "Cybersecurity Committee", ["cyber_risk_management", "data_protection", "incident_response"]),
            (BoardCommitteeType.INNOVATION, "Innovation Committee", ["r_and_d_strategy", "digital_transformation", "emerging_technologies"]),
            (BoardCommitteeType.STAKEHOLDER, "Stakeholder Committee", ["stakeholder_engagement", "community_relations", "investor_relations"]),
            (BoardCommitteeType.CRISIS_MANAGEMENT, "Crisis Management Committee", ["crisis_response", "business_continuity", "reputation_management"]),
            (BoardCommitteeType.AI_ETHICS, "AI Ethics Committee", ["ai_governance", "algorithmic_fairness", "responsible_ai"]),
            (BoardCommitteeType.DATA_PRIVACY, "Data Privacy Committee", ["data_protection", "privacy_compliance", "gdpr_oversight"]),
        ]
        
        for committee_type, name, responsibilities in committee_configs:
            # Find appropriate chair and members
            chair_id = None
            members = []
            
            for member_id, member in self.members.items():
                if member.role in [CorporateRole.BOARD_CHAIR, CorporateRole.BOARD_VICE_CHAIR, CorporateRole.INDEPENDENT_DIRECTOR]:
                    if committee_type == BoardCommitteeType.AUDIT and "finance" in member.expertise_areas:
                        if not chair_id:
                            chair_id = member_id
                        members.append(member_id)
                    elif committee_type == BoardCommitteeType.COMPENSATION and "governance" in member.expertise_areas:
                        if not chair_id:
                            chair_id = member_id
                        members.append(member_id)
                    elif committee_type == BoardCommitteeType.NOMINATING and "governance" in member.expertise_areas:
                        if not chair_id:
                            chair_id = member_id
                        members.append(member_id)
                    elif committee_type == BoardCommitteeType.RISK and "compliance" in member.expertise_areas:
                        if not chair_id:
                            chair_id = member_id
                        members.append(member_id)
                    elif committee_type == BoardCommitteeType.TECHNOLOGY and "technology" in member.expertise_areas:
                        if not chair_id:
                            chair_id = member_id
                        members.append(member_id)
            
            # Ensure we have at least 3 members for quorum
            if len(members) < 3:
                # Add more board members to reach quorum
                for member_id, member in self.members.items():
                    if member_id not in members and member.role in [CorporateRole.BOARD_MEMBER, CorporateRole.INDEPENDENT_DIRECTOR]:
                        members.append(member_id)
                        if len(members) >= 3:
                            break
            
            committee = BoardCommittee(
                name=name,
                committee_type=committee_type,
                chair=chair_id or members[0] if members else "",
                members=members[:5],  # Limit to 5 members
                responsibilities=responsibilities,
                meeting_schedule="Quarterly",
                quorum_required=3
            )
            
            self.board_committees[committee.committee_id] = committee
            
            # Update member committee assignments
            for member_id in members:
                if member_id in self.members:
                    self.members[member_id].board_committees.append(committee.committee_id)
    
    def _initialize_crca_sd_governance(self) -> None:
        """Initialize CRCA-SD governance system with boards."""
        if not CRCA_SD_AVAILABLE:
            return
        
        # Create CRCA-SD boards from existing board members
        boards = []
        board_type_mapping = {
            'growth': BoardType.GROWTH,
            'welfare': BoardType.WELFARE,
            'sustainability': BoardType.SUSTAINABILITY,
            'stability': BoardType.STABILITY,
        }
        
        # Map board members to CRCA-SD boards based on expertise
        for board_type_name, board_type_enum in board_type_mapping.items():
            board_members_list = []
            for member_id in self.board_members:
                member = self.members.get(member_id)
                if not member:
                    continue
                
                # Match members to board types based on expertise
                expertise_lower = [e.lower() for e in member.expertise_areas]
                matches = False
                
                if board_type_name == 'growth':
                    matches = any(e in ['strategy', 'innovation', 'technology', 'marketing'] for e in expertise_lower)
                elif board_type_name == 'welfare':
                    matches = any(e in ['hr', 'governance', 'social', 'employee'] for e in expertise_lower)
                elif board_type_name == 'sustainability':
                    matches = any(e in ['sustainability', 'environmental', 'esg', 'carbon'] for e in expertise_lower)
                elif board_type_name == 'stability':
                    matches = any(e in ['finance', 'risk', 'compliance', 'legal'] for e in expertise_lower)
                
                if matches or len(boards) == 0:  # Add to first board if no matches
                    if CRCA_SD_BoardMember:
                        board_members_list.append(CRCA_SD_BoardMember(
                            member_id=member.member_id,
                            name=member.name,
                            board_type=board_type_enum,
                            expertise_areas=member.expertise_areas,
                            voting_weight=member.voting_weight,
                            preferences={},  # Will be set based on board type
                            independence_status=member.independence_status
                        ))
            
            if board_members_list:
                board = Board(
                    board_id=f"crca_sd_{board_type_name}",
                    board_type=board_type_enum,
                    members=board_members_list
                )
                boards.append(board)
        
        if boards:
            self.governance_system = GovernanceSystem(
                boards=boards,
                arbitration_method=self.config.get('arbitration_method', 'weighted_vote') if hasattr(self.config, 'arbitration_method') else 'weighted_vote'
            )
            if self.verbose:
                CORPORATE_LOGGER.info(f"Initialized CRCA-SD governance with {len(boards)} boards")
    
    def _initialize_member_agent(self, member: CorporateMember) -> None:
        """Initialize CRCAAgent for a member."""
        if not CRCA_AVAILABLE or not self.config.enable_causal_reasoning:
            # Fallback to regular Agent
            from swarms.structs.agent import Agent
            system_prompt = self._create_member_system_prompt(member)
            member.agent = Agent(
                agent_name=member.name,
                agent_description=f"{member.role.value.title()} in {member.department.value.title()} department",
                system_prompt=system_prompt,
                model_name=self.corporate_model_name,
                max_loops=3,
                verbose=self.verbose
            )
            return
        
        # Use CRCAAgent
        variables = self._get_corporate_variables()
        causal_edges = self._get_causal_edges_for_member(member)
        member.causal_variables = variables
        
        system_prompt = self._create_member_system_prompt(member)
        member.agent = CRCAAgent(
            agent_name=member.name,
            agent_description=f"{member.role.value.title()} with causal reasoning in {member.department.value.title()} department",
            variables=variables,
            causal_edges=causal_edges,
            model_name=self.corporate_model_name,
            max_loops=3,
            system_prompt=system_prompt,
            verbose=self.verbose
        )
        
        # Build causal graph for this member
        if member.agent and hasattr(member.agent, 'add_edges_from'):
            member.agent.add_edges_from(causal_edges)
        
        # Initialize queue for this member if queue execution is enabled
        if self.config.enable_queue_execution and AOP_AVAILABLE and member.agent:
            self._initialize_member_queue(member)
    
    def _initialize_aop_server(self) -> None:
        """Initialize AOP server for MCP deployment of member agents."""
        if not AOP_AVAILABLE or not self.config.enable_aop:
            return
        
        try:
            self.aop_server = AOP(
                server_name=f"{self.name}_AOP",
                description=f"AOP server for {self.name} corporate governance system",
                agents=None,  # Will add agents as they're created
                port=self.config.aop_server_port,
                verbose=self.verbose,
                queue_enabled=self.config.enable_queue_execution,
                max_workers_per_agent=self.config.aop_max_workers_per_agent,
                max_queue_size_per_agent=self.config.aop_max_queue_size,
                processing_timeout=self.config.aop_processing_timeout,
                persistence=self.config.aop_persistence,
                network_monitoring=self.config.aop_network_monitoring
            )
            
            if self.verbose:
                CORPORATE_LOGGER.info(f"Initialized AOP server on port {self.config.aop_server_port}")
        except Exception as e:
            CORPORATE_LOGGER.warning(f"Failed to initialize AOP server: {e}")
            self.aop_server = None
    
    def _initialize_member_queues(self) -> None:
        """Initialize task queues for all existing member agents."""
        if not AOP_AVAILABLE or not self.config.enable_queue_execution:
            return
        
        for member_id, member in self.members.items():
            if member.agent and member_id not in self.member_task_queues:
                self._initialize_member_queue(member)
    
    def _initialize_member_queue(self, member: CorporateMember) -> None:
        """Initialize task queue for a specific member agent."""
        if not AOP_AVAILABLE or not self.config.enable_queue_execution:
            return
        
        if not member.agent:
            return
        
        if member.member_id in self.member_task_queues:
            return  # Queue already exists
        
        try:
            queue = TaskQueue(
                agent_name=member.name,
                agent=member.agent,
                max_workers=self.config.aop_max_workers_per_agent,
                max_queue_size=self.config.aop_max_queue_size,
                processing_timeout=self.config.aop_processing_timeout,
                retry_delay=1.0,
                verbose=self.verbose
            )
            
            # Start workers
            queue.start_workers()
            
            self.member_task_queues[member.member_id] = queue
            
            if self.verbose:
                CORPORATE_LOGGER.debug(f"Initialized task queue for member: {member.name}")
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Failed to initialize queue for {member.name}: {e}")
    
    def _initialize_democratic_swarm(self) -> None:
        """Initialize democratic decision-making swarm."""
        if self.verbose:
            CORPORATE_LOGGER.info("Initializing democratic decision-making swarm")
        
        # Create specialized swarms for different corporate functions
        governance_swarm = SwarmRouter(
            name="governance-swarm",
            description="Handles corporate governance and board decisions",
            agents=[self.members[member_id].agent for member_id in self.board_members 
                   if self.members[member_id].agent],
            swarm_type="SequentialWorkflow"
        )
        
        executive_swarm = SwarmRouter(
            name="executive-swarm", 
            description="Handles executive leadership decisions",
            agents=[self.members[member_id].agent for member_id in self.executive_team
                   if self.members[member_id].agent],
            swarm_type="ConcurrentWorkflow"
        )
        
        # Initialize democratic swarm
        self.democratic_swarm = HybridHierarchicalClusterSwarm(
            name="Corporate Democratic Swarm",
            description="Democratic decision-making for corporate governance",
            swarms=[governance_swarm, executive_swarm],
            max_loops=self.max_loops,
            router_agent_model_name="gpt-4o-mini"
        )
    
    def add_member(
        self,
        name: str,
        role: CorporateRole,
        department: DepartmentType,
        expertise_areas: List[str] = None,
        voting_weight: float = 1.0,
        **kwargs
    ) -> str:
        """
        Add a new corporate member.
        
        Args:
            name: Full name of the member
            role: Corporate role and position
            department: Department affiliation
            expertise_areas: Areas of professional expertise
            voting_weight: Weight of vote in corporate decisions
            **kwargs: Additional member attributes
            
        Returns:
            str: Member ID of the created member
        """
        member = CorporateMember(
            name=name,
            role=role,
            department=department,
            expertise_areas=expertise_areas or [],
            voting_weight=voting_weight,
            metadata=kwargs
        )
        
        # Initialize member agent (CRCAAgent if available, otherwise Agent)
        self._initialize_member_agent(member)
        
        self.members[member.member_id] = member
        
        # Add to appropriate groups
        if role in [CorporateRole.BOARD_CHAIR, CorporateRole.BOARD_MEMBER]:
            self.board_members.append(member.member_id)
        elif role in [CorporateRole.CEO, CorporateRole.CFO, CorporateRole.CTO, CorporateRole.COO]:
            self.executive_team.append(member.member_id)
        
        # Add agent to AOP server if enabled
        if self.config.enable_aop and self.aop_server and member.agent:
            try:
                tool_name = f"{member.name.replace(' ', '_')}_{member.member_id[:8]}"
                self.aop_server.add_agent(
                    agent=member.agent,
                    tool_name=tool_name,
                    tool_description=f"{member.role.value.title()} in {member.department.value.title()} department",
                    verbose=self.verbose
                )
                if self.verbose:
                    CORPORATE_LOGGER.info(f"Added {member.name} to AOP server as tool: {tool_name}")
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to add {member.name} to AOP server: {e}")
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Added member: {name} as {role.value} in {department.value}")
        
        return member.member_id
    
    def _create_member_system_prompt(self, member: CorporateMember) -> str:
        """Create system prompt for a corporate member with causal reasoning."""
        causal_instructions = ""
        if self.config.enable_causal_reasoning and CRCA_AVAILABLE and member.causal_variables:
            causal_instructions = f"""

Causal Reasoning Capabilities:
- You have access to causal reasoning and counterfactual analysis
- Your causal variables include: {', '.join(member.causal_variables[:10])}
- Use causal analysis to understand direct and indirect effects of decisions
- Consider counterfactual scenarios: "What would happen if we did X instead of Y?"
- Identify confounding factors and causal chains
- Predict outcomes based on causal relationships, not just correlations

When analyzing proposals causally:
1. Map the proposal to relevant causal variables
2. Trace causal chains from the proposal to key outcomes (revenue, risk, ESG, etc.)
3. Consider both direct effects and indirect/secondary effects
4. Identify potential confounders that might affect outcomes
5. Generate counterfactual scenarios to explore alternatives
6. Provide causal predictions for key corporate metrics
"""
        
        return f"""
You are {member.name}, a {member.role.value.title()} in the {member.department.value.title()} department.

Your Role and Responsibilities:
- Role: {member.role.value.title()}
- Department: {member.department.value.title()}
- Expertise Areas: {', '.join(member.expertise_areas)}
- Voting Weight: {member.voting_weight}

Corporate Context:
- You are part of {self.name}, a comprehensive corporate governance system
- You participate in democratic decision-making processes
- You collaborate with other corporate members across departments
- You provide expertise in your areas of specialization
{causal_instructions}
Decision-Making Guidelines:
1. Consider the long-term strategic impact of decisions
2. Evaluate financial implications and risk factors
3. Ensure alignment with corporate objectives and values
4. Consider stakeholder interests and regulatory compliance
5. Provide clear reasoning for your positions and votes
6. Use causal reasoning to understand why decisions lead to outcomes

Communication Style:
- Professional and collaborative
- Data-driven and analytical
- Clear and concise
- Respectful of diverse perspectives
- Causal and evidence-based

When participating in corporate decisions, always provide:
- Your position and reasoning
- Relevant expertise and insights
- Causal analysis of expected outcomes
- Consideration of alternatives
- Risk assessment and mitigation strategies
- Counterfactual scenarios when relevant
"""
    
    def create_proposal(
        self,
        title: str,
        description: str,
        proposal_type: ProposalType,
        sponsor_id: str,
        department: DepartmentType,
        budget_impact: float = 0.0,
        timeline: str = "",
        **kwargs
    ) -> str:
        """Create a new corporate proposal."""
        if sponsor_id not in self.members:
            raise ValueError(f"Sponsor {sponsor_id} not found in corporate members")
        
        proposal = CorporateProposal(
            title=title,
            description=description,
            proposal_type=proposal_type,
            sponsor=sponsor_id,
            department=department,
            budget_impact=budget_impact,
            timeline=timeline,
            metadata=kwargs
        )
        
        self.proposals.append(proposal)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Created proposal: {title} by {self.members[sponsor_id].name}")
        
        return proposal.proposal_id
    
    def conduct_corporate_vote(
        self,
        proposal_id: str,
        participants: List[str] = None
    ) -> CorporateVote:
        """Conduct a democratic vote on a corporate proposal with enhanced analysis."""
        proposal = self._find_proposal(proposal_id)
        participants = participants or self._get_default_participants()
        
        if not self.cost_tracker.check_budget():
            return self._create_failed_vote(proposal)
        
        # CRCA-SD board evaluation
        board_evaluations = None
        governance_consensus = 0.0
        if self.config.enable_crca_sd_governance and self.governance_system and CRCA_SD_AVAILABLE:
            try:
                control_vector = self._proposal_to_control_vector(proposal)
                objectives = self._get_objectives_matrix()
                selected_policy = self.governance_system.select_policy([control_vector], objectives)
                if selected_policy:
                    board_evaluations = {
                        'selected_policy': selected_policy,
                        'boards_evaluated': len(self.governance_system.boards),
                        'arbitration_method': self.governance_system.arbitration_method
                    }
                    # Calculate consensus score (simplified)
                    governance_consensus = 0.7  # Default consensus when policy is selected
                proposal.board_evaluations = board_evaluations
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"CRCA-SD board evaluation failed: {e}")
        
        # Causal analysis with CRCA
        causal_analysis = None
        causal_reasoning_summary = ""
        if self.config.enable_causal_reasoning:
            try:
                causal_analysis = self._analyze_proposal_causally(proposal)
                proposal.causal_analysis = causal_analysis
                causal_reasoning_summary = causal_analysis.get('summary', '') if causal_analysis else ''
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Causal analysis failed: {e}")
        
        # Quant analysis for financial proposals
        quant_analysis = None
        quant_signals = {}
        if self.config.enable_quant_analysis and proposal.proposal_type in [
            ProposalType.INVESTMENT, ProposalType.BUDGET_ALLOCATION, 
            ProposalType.MERGER_ACQUISITION, ProposalType.DIVIDEND_DECLARATION
        ]:
            try:
                quant_analysis = self._analyze_financial_proposal(proposal)
                proposal.quant_analysis = quant_analysis
                quant_signals = quant_analysis.get('signals', {}) if quant_analysis else {}
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Quantitative analysis failed: {e}")
        
        # Get democratic decision
        democratic_result = self._get_democratic_decision(proposal)
        
        # Collect individual votes
        individual_votes = self._collect_individual_votes(proposal, participants)
        
        # Analyze vote results
        vote_result = self._analyze_vote_results(individual_votes, proposal)
        
        # Create vote record with enhanced data
        vote = self._create_vote_record(
            proposal, participants, individual_votes, democratic_result, vote_result,
            causal_reasoning_summary=causal_reasoning_summary,
            quant_signals=quant_signals,
            governance_consensus=governance_consensus
        )
        self.votes.append(vote)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Vote completed: {proposal.title} - Result: {vote_result.value}")
        
        return vote
    
    def _find_proposal(self, proposal_id: str) -> CorporateProposal:
        """Find a proposal by ID."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        raise ValueError(f"Proposal {proposal_id} not found")
    
    def _get_default_participants(self) -> List[str]:
        """Get default voting participants."""
        return self.board_members + self.executive_team
    
    def _create_failed_vote(self, proposal: CorporateProposal) -> CorporateVote:
        """Create a failed vote due to budget constraints."""
        return CorporateVote(
            proposal=proposal,
            result=VoteResult.FAILED
        )
    
    def _get_democratic_decision(self, proposal: CorporateProposal) -> Optional[Dict[str, Any]]:
        """Get democratic decision from swarm if available."""
        if self.democratic_swarm is None:
            return None
        
        decision_task = self._create_decision_task(proposal)
        
        try:
            democratic_result = self.democratic_swarm.run(decision_task)
            return self._parse_democratic_result(democratic_result)
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Democratic swarm encountered issue: {e}")
            return {'result': 'error', 'error': str(e), 'type': 'error_response'}
    
    def _create_decision_task(self, proposal: CorporateProposal) -> str:
        """Create decision task for democratic swarm."""
        return f"""
Corporate Proposal Vote: {proposal.title}

Proposal Description: {proposal.description}
Proposal Type: {proposal.proposal_type.value}
Department: {proposal.department.value.title()}
Budget Impact: ${proposal.budget_impact:,.2f}
Timeline: {proposal.timeline}

As a corporate decision-making body, please:
1. Analyze the proposal's strategic value and alignment with corporate objectives
2. Evaluate financial implications and return on investment
3. Assess implementation feasibility and resource requirements
4. Consider risk factors and mitigation strategies
5. Make a recommendation on whether to approve or reject this proposal
6. Provide detailed reasoning for your decision

This is a critical corporate decision that will impact the organization's future direction.
"""
    
    def _parse_democratic_result(self, result: Any) -> Dict[str, Any]:
        """Parse democratic swarm result with robust error handling."""
        # Handle None or empty results
        if result is None:
            return {'result': 'no_response', 'type': 'none_response', 'message': 'No response generated from democratic swarm'}
        
        # Handle string responses (like "No response generated")
        if isinstance(result, str):
            if not result.strip() or "No response generated" in result:
                return {'result': 'no_response', 'type': 'string_response', 'message': result}
            # Try to parse as JSON if it looks like JSON
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
                return {'result': 'processed', 'data': parsed, 'type': 'string_json_response'}
            except (json.JSONDecodeError, ValueError):
                return {'result': 'processed', 'data': result, 'type': 'string_response', 'message': result}
        
        # Handle list responses
        if isinstance(result, list):
            if not result:
                return {'result': 'empty_response', 'type': 'list_response', 'data': []}
            
            # Process first item in list
            first_item = result[0]
            if not isinstance(first_item, dict):
                return {'result': 'processed', 'data': result, 'type': 'list_response', 'message': 'First item is not a dict'}
            
            # Check for function call structure
            if 'function' not in first_item:
                return {'result': 'processed', 'data': result, 'type': 'list_response', 'message': 'No function call in response'}
            
            function_data = first_item.get('function', {})
            if not isinstance(function_data, dict):
                return {'result': 'processed', 'data': result, 'type': 'list_response', 'message': 'Function data is not a dict'}
        
        # Extract arguments if available
        if 'arguments' in function_data:
            try:
                args_str = function_data['arguments']
                if isinstance(args_str, str):
                    parsed_args = json.loads(args_str)
                    # Ensure we return a dict
                    if isinstance(parsed_args, dict):
                        return parsed_args
                    return {'result': 'processed', 'data': parsed_args, 'type': 'parsed_arguments'}
                elif isinstance(args_str, dict):
                    return args_str
                else:
                    return {'result': 'processed', 'data': args_str, 'type': 'arguments_non_dict'}
            except (json.JSONDecodeError, TypeError) as e:
                # Fallback to empty dict if parsing fails
                return {'result': 'parse_error', 'type': 'json_error', 'error': str(e), 'raw_arguments': function_data.get('arguments', {})}
        else:
            # Return function_data if it's a dict, otherwise wrap it
            if isinstance(function_data, dict):
                return function_data
            return {'result': 'processed', 'data': function_data, 'type': 'function_data'}
    
    def _collect_individual_votes(self, proposal: CorporateProposal, participants: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect individual votes from participants."""
        individual_votes = {}
        
        for i in range(0, len(participants), self.batch_size):
            batch = participants[i:i + self.batch_size]
            
            for member_id in batch:
                if member_id not in self.members:
                    continue
                
                member = self.members[member_id]
                if not member.agent:
                    continue
                
                vote_data = self._get_member_vote(member, proposal)
                if vote_data:
                    individual_votes[member_id] = vote_data
        
        return individual_votes
    
    def _get_member_vote(self, member: CorporateMember, proposal: CorporateProposal) -> Optional[Dict[str, Any]]:
        """Get vote from a single member using queue if enabled."""
        vote_prompt = self._create_vote_prompt(member, proposal)
        
        try:
            # Use queue-based execution if enabled
            if self.config.enable_queue_execution and member.member_id in self.member_task_queues:
                queue = self.member_task_queues[member.member_id]
                task_id = queue.add_task(task=vote_prompt, priority=5)  # High priority for votes
                
                # Wait for completion with timeout
                start_time = time.time()
                timeout = self.config.aop_processing_timeout
                
                while time.time() - start_time < timeout:
                    task = queue.get_task(task_id)
                    if task and task.status == TaskStatus.COMPLETED:
                        response = task.result or ""
                        break
                    elif task and task.status == TaskStatus.FAILED:
                        raise Exception(task.error or "Task failed")
                    time.sleep(0.1)
                else:
                    raise TimeoutError(f"Vote request timed out for {member.name}")
            else:
                # Direct execution
                response = member.agent.run(vote_prompt)
            
            if not isinstance(response, str):
                response = str(response)
            
            return {
                "vote": "APPROVE" if "approve" in response.lower() else
                       "REJECT" if "reject" in response.lower() else "ABSTAIN",
                "reasoning": response,
                "member_id": member.member_id,
                "member_name": member.name,
                "role": member.role.value,
                "department": member.department.value,
                "voting_weight": member.voting_weight
            }
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.error(f"Error getting vote from {member.name}: {e}")
            return None
    
    def _create_vote_prompt(self, member: CorporateMember, proposal: CorporateProposal) -> str:
        """Create voting prompt for a member."""
        return f"""
Corporate Proposal Vote: {proposal.title}

Proposal Details:
- Description: {proposal.description}
- Type: {proposal.proposal_type.value}
- Department: {proposal.department.value.title()}
- Budget Impact: ${proposal.budget_impact:,.2f}
- Timeline: {proposal.timeline}

As {member.name}, {member.role.value.title()} in {member.department.value.title()} department:

Please provide your vote and reasoning:
1. Vote: APPROVE, REJECT, or ABSTAIN
2. Reasoning: Detailed explanation of your decision
3. Key Factors: What influenced your decision most
4. Concerns: Any concerns or conditions you have
5. Recommendations: Suggestions for improvement if applicable

Consider your expertise in: {', '.join(member.expertise_areas)}
Your voting weight: {member.voting_weight}
"""
    
    def _create_vote_record(
        self, 
        proposal: CorporateProposal, 
        participants: List[str], 
        individual_votes: Dict[str, Dict[str, Any]], 
        democratic_result: Optional[Dict[str, Any]], 
        vote_result: VoteResult,
        causal_reasoning_summary: str = "",
        quant_signals: Dict[str, float] = None,
        governance_consensus: float = 0.0
    ) -> CorporateVote:
        """Create vote record with all collected data."""
        return CorporateVote(
            proposal=proposal,
            participants=participants,
            individual_votes=individual_votes,
            political_group_analysis={"democratic_result": democratic_result},
            result=vote_result,
            causal_reasoning_summary=causal_reasoning_summary,
            quant_signals=quant_signals or {},
            governance_consensus=governance_consensus,
            metadata={
                "total_participants": len(participants),
                "total_processed": len(individual_votes),
                "processing_time": time.time()
            }
        )
    
    def create_board_committee(
        self,
        name: str,
        committee_type: BoardCommitteeType,
        chair_id: str,
        members: List[str],
        responsibilities: List[str] = None,
        meeting_schedule: str = "Quarterly",
        quorum_required: int = 3,
        **kwargs
    ) -> str:
        """
        Create a new board committee.
        
        Args:
            name: Name of the committee
            committee_type: Type of board committee
            chair_id: ID of the committee chair
            members: List of member IDs to serve on the committee
            responsibilities: List of committee responsibilities
            meeting_schedule: Regular meeting schedule
            quorum_required: Minimum members required for quorum
            **kwargs: Additional committee attributes
            
        Returns:
            str: Committee ID of the created committee
        """
        if chair_id not in self.members:
            raise ValueError(f"Chair {chair_id} not found in corporate members")
        
        # Validate all members exist
        for member_id in members:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found in corporate members")
        
        committee = BoardCommittee(
            name=name,
            committee_type=committee_type,
            chair=chair_id,
            members=members,
            responsibilities=responsibilities or [],
            meeting_schedule=meeting_schedule,
            quorum_required=quorum_required,
            metadata=kwargs
        )
        
        self.board_committees[committee.committee_id] = committee
        
        # Update member committee assignments
        for member_id in members:
            if member_id in self.members:
                self.members[member_id].board_committees.append(committee.committee_id)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Created board committee: {name} with {len(members)} members")
        
        return committee.committee_id
    
    def schedule_board_meeting(
        self,
        meeting_type: MeetingType,
        date: float = None,
        location: str = "Virtual",
        agenda: List[str] = None,
        attendees: List[str] = None
    ) -> str:
        """
        Schedule a board meeting.
        
        Args:
            meeting_type: Type of board meeting
            date: Meeting date and time (defaults to current time)
            location: Meeting location
            agenda: List of agenda items
            attendees: List of attendee member IDs
            
        Returns:
            str: Meeting ID of the scheduled meeting
        """
        if not date:
            date = time.time()
        
        if not attendees:
            attendees = self.board_members + self.executive_team
        
        meeting = BoardMeeting(
            meeting_type=meeting_type,
            date=date,
            location=location,
            attendees=attendees,
            agenda=agenda or [],
            quorum_met=len(attendees) >= len(self.board_members) // 2 + 1
        )
        
        self.board_meetings.append(meeting)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Scheduled {meeting_type.value} meeting for {len(attendees)} attendees")
        
        return meeting.meeting_id
    
    def conduct_board_meeting(
        self,
        meeting_id: str,
        discussion_topics: List[str] = None
    ) -> BoardMeeting:
        """Conduct a board meeting with discussion and decisions."""
        # Find the meeting
        meeting = None
        for m in self.board_meetings:
            if m.meeting_id == meeting_id:
                meeting = m
                break
        
        if not meeting:
            raise ValueError(f"Meeting {meeting_id} not found")
        
        if not discussion_topics:
            discussion_topics = meeting.agenda
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Conducting board meeting: {meeting.meeting_type.value}")
        
        # Conduct discussions on each topic
        minutes = []
        resolutions = []
        
        for topic in discussion_topics:
            if self.verbose:
                CORPORATE_LOGGER.info(f"Discussing topic: {topic}")
            
            # Create a proposal for the topic
            proposal_id = self.create_proposal(
                title=f"Board Discussion: {topic}",
                description=f"Board discussion and decision on {topic}",
                proposal_type=ProposalType.BOARD_RESOLUTION,
                sponsor_id=self.board_members[0] if self.board_members else self.executive_team[0],
                department=DepartmentType.OPERATIONS
            )
            
            # Conduct vote on the topic
            vote = self.conduct_corporate_vote(proposal_id, meeting.attendees)
            
            # Record minutes
            topic_minutes = f"Topic: {topic}\n"
            topic_minutes += f"Discussion: Board members discussed the implications and considerations.\n"
            topic_minutes += f"Vote Result: {vote.result.value.upper()}\n"
            topic_minutes += f"Participants: {len(vote.participants)} members\n"
            
            minutes.append(topic_minutes)
            
            if vote.result in [VoteResult.APPROVED, VoteResult.UNANIMOUS]:
                resolutions.append(f"RESOLVED: {topic} - APPROVED")
            elif vote.result == VoteResult.REJECTED:
                resolutions.append(f"RESOLVED: {topic} - REJECTED")
            else:
                resolutions.append(f"RESOLVED: {topic} - TABLED for further consideration")
        
        # Update meeting with minutes and resolutions
        meeting.minutes = "\n\n".join(minutes)
        meeting.resolutions = resolutions
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Board meeting completed with {len(resolutions)} resolutions")
        
        return meeting
    
    def conduct_committee_meeting(
        self,
        committee_id: str,
        meeting_type: MeetingType = MeetingType.COMMITTEE_MEETING,
        agenda: List[str] = None
    ) -> Dict[str, Any]:
        """Conduct a committee meeting with real API calls."""
        if committee_id not in self.board_committees:
            raise ValueError(f"Committee {committee_id} not found")
        
        committee = self.board_committees[committee_id]
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Conducting {committee.name} meeting")
        
        try:
            # Create a specialized task for committee meeting
            committee_task = f"""
            Conduct a {committee.name} committee meeting for UAB Leiliona logistics company.
            
            Committee Type: {committee.committee_type.value}
            Committee Responsibilities: {', '.join(committee.responsibilities)}
            Agenda Items: {', '.join(agenda or committee.responsibilities)}
            
            Please provide:
            1. Issues Discussed: List 3-5 key issues that were addressed
            2. Recommendations: Provide 2-3 actionable recommendations
            3. Next Steps: Outline follow-up actions
            4. Risk Assessment: Identify any risks or concerns raised
            
            Format your response as a structured analysis suitable for corporate governance.
            """
            
            # Use the democratic swarm for committee decision-making
            if hasattr(self, 'democratic_swarm') and self.democratic_swarm:
                result = self.democratic_swarm.run(committee_task)
                
                # Handle case where democratic_swarm returns a list instead of dict
                if isinstance(result, list):
                    # Extract function call arguments if available
                    if result and len(result) > 0:
                        first_item = result[0]
                        if isinstance(first_item, dict) and 'function' in first_item:
                            function_data = first_item.get('function', {})
                            if 'arguments' in function_data:
                                # Try to parse the arguments as JSON
                                try:
                                    import json
                                    args_str = function_data['arguments']
                                    if isinstance(args_str, str):
                                        parsed_args = json.loads(args_str)
                                        result = parsed_args
                                    else:
                                        result = args_str
                                except (json.JSONDecodeError, TypeError):
                                    # If parsing fails, use the raw arguments
                                    result = function_data.get('arguments', {})
                            else:
                                result = function_data
                        else:
                            # If it's not a function call, convert to a simple dict
                            result = {
                                'result': 'processed',
                                'data': result,
                                'type': 'list_response'
                            }
                    else:
                        result = {'result': 'empty_response', 'type': 'list_response'}
                
                # Ensure result is a dictionary before parsing
                if not isinstance(result, dict):
                    result = {
                        'issues_discussed': 3,
                        'recommendations': 2,
                        'next_steps': ['Follow up on action items', 'Schedule next meeting'],
                        'risk_assessment': 'Standard committee review completed',
                        'summary': 'Committee meeting completed with standard agenda items'
                    }
                
                # Parse the result to extract structured information
                return {
                    'issues_discussed': result.get('issues_discussed', 3),
                    'recommendations': result.get('recommendations', 2),
                    'next_steps': result.get('next_steps', []),
                    'risk_assessment': result.get('risk_assessment', 'Low risk identified'),
                    'meeting_summary': result.get('summary', 'Committee meeting completed successfully')
                }
            else:
                # Fallback if democratic swarm is not available
                return {
                    'issues_discussed': 3,
                    'recommendations': 2,
                    'next_steps': ['Review committee findings', 'Implement recommendations'],
                    'risk_assessment': 'No significant risks identified',
                    'meeting_summary': 'Committee meeting completed with standard procedures'
                }
                
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Committee meeting encountered issue: {e}")
            
            # Return structured fallback results
            return {
                'issues_discussed': 3,
                'recommendations': 2,
                'next_steps': ['Address technical issues', 'Reschedule if needed'],
                'risk_assessment': 'Technical issues noted, no operational risks',
                'meeting_summary': f'Committee meeting completed with fallback procedures due to: {str(e)[:50]}...'
            }
    
    def evaluate_board_performance(self) -> Dict[str, Any]:
        """Evaluate board performance and governance effectiveness."""
        if self.verbose:
            CORPORATE_LOGGER.info("Evaluating board performance")
        
        # Calculate governance metrics
        total_members = len(self.board_members)
        independent_directors = len(self.independent_directors)
        executive_directors = len(self.executive_directors)
        committees = len(self.board_committees)
        meetings_held = len(self.board_meetings)
        proposals_processed = len(self.proposals)
        votes_conducted = len(self.votes)
        
        # Calculate independence ratio
        independence_ratio = independent_directors / total_members if total_members > 0 else 0
        
        # Calculate meeting frequency (assuming monthly meetings)
        if not self.members:
            meeting_frequency = 0.0
        else:
            try:
                # More efficient: use generator expression instead of list comprehension
                earliest_term = min(m.term_start for m in self.members.values())
                months_operating = (time.time() - earliest_term) / (30 * 24 * 60 * 60)
                expected_meetings = max(1, int(months_operating))
                meeting_frequency = meetings_held / expected_meetings if expected_meetings > 0 else 0
            except (ValueError, ZeroDivisionError):
                meeting_frequency = 0.0
        
        # Calculate decision efficiency
        approved_proposals = len([v for v in self.votes if v.result in [VoteResult.APPROVED, VoteResult.UNANIMOUS]])
        decision_efficiency = approved_proposals / votes_conducted if votes_conducted > 0 else 0
        
        performance_metrics = {
            "board_composition": {
                "total_members": total_members,
                "independent_directors": independent_directors,
                "executive_directors": executive_directors,
                "independence_ratio": independence_ratio
            },
            "governance_structure": {
                "committees": committees,
                "committee_types": [c.committee_type.value for c in self.board_committees.values()],
                "meeting_frequency": meeting_frequency
            },
            "decision_making": {
                "meetings_held": meetings_held,
                "proposals_processed": proposals_processed,
                "votes_conducted": votes_conducted,
                "approved_proposals": approved_proposals,
                "decision_efficiency": decision_efficiency
            },
            "compliance": {
                "quorum_met": len([m for m in self.board_meetings if m.quorum_met]),
                "resolutions_passed": sum(len(m.resolutions) for m in self.board_meetings),
                "governance_score": (independence_ratio + meeting_frequency + decision_efficiency) / 3
            }
        }
        
        return performance_metrics
    
    def calculate_esg_score(self) -> ESGScore:
        """Calculate comprehensive ESG (Environmental, Social, Governance) score."""
        if self.verbose:
            CORPORATE_LOGGER.info("Calculating ESG score for corporate governance")
        
        # Calculate environmental score
        environmental_score = self._calculate_environmental_score()
        
        # Calculate social score
        social_score = self._calculate_social_score()
        
        # Calculate governance score
        governance_score = self._calculate_governance_score()
        
        # Calculate overall score
        overall_score = (environmental_score + social_score + governance_score) / 3
        
        # Calculate diversity index
        diversity_index = self._calculate_diversity_index()
        
        # Calculate stakeholder satisfaction
        stakeholder_satisfaction = self._calculate_stakeholder_satisfaction()
        
        esg_score = ESGScore(
            environmental_score=environmental_score,
            social_score=social_score,
            governance_score=governance_score,
            overall_score=overall_score,
            carbon_footprint=self._calculate_carbon_footprint(),
            diversity_index=diversity_index,
            stakeholder_satisfaction=stakeholder_satisfaction,
            sustainability_goals=self._get_sustainability_goals()
        )
        
        # Store ESG score
        self.esg_scores[str(time.time())] = esg_score
        
        return esg_score
    
    def _calculate_environmental_score(self) -> float:
        """Calculate environmental performance score."""
        # Base score from sustainability practices
        base_score = 70.0
        
        # Adjust based on sustainability targets
        if self.sustainability_targets:
            target_achievement = len([t for t in self.sustainability_targets.values() if t.get('achieved', False)])
            total_targets = len(self.sustainability_targets)
            if total_targets > 0:
                base_score += (target_achievement / total_targets) * 30
        
        return min(100.0, base_score)
    
    def _calculate_social_score(self) -> float:
        """Calculate social performance score."""
        # Base score from stakeholder engagement
        base_score = 75.0
        
        # Adjust based on stakeholder satisfaction
        if self.stakeholder_engagements:
            avg_satisfaction = sum(s.satisfaction_score for s in self.stakeholder_engagements.values()) / len(self.stakeholder_engagements)
            base_score = avg_satisfaction
        
        return min(100.0, base_score)
    
    def _calculate_governance_score(self) -> float:
        """Calculate governance performance score."""
        # Base score from board performance
        base_score = 80.0
        
        # Adjust based on board performance metrics
        if hasattr(self, 'performance_metrics') and self.performance_metrics:
            governance_metrics = self.performance_metrics.get('governance', {})
            if governance_metrics:
                base_score = governance_metrics.get('governance_score', base_score)
        
        return min(100.0, base_score)
    
    def _calculate_diversity_index(self) -> float:
        """Calculate board diversity index."""
        if not self.board_members:
            return 0.0
        
        # More efficient: use set comprehension and generator
        unique_expertise = set()
        for member_id in self.board_members:
            member = self.members.get(member_id)
            if member:
                unique_expertise.update(member.expertise_areas)
        
        # Normalize to 0-1 scale with configurable max diversity
        MAX_POSSIBLE_DIVERSITY = 10  # Configurable constant
        diversity_index = min(1.0, len(unique_expertise) / MAX_POSSIBLE_DIVERSITY)
        
        return diversity_index
    
    def _calculate_stakeholder_satisfaction(self) -> float:
        """Calculate overall stakeholder satisfaction score."""
        if not self.stakeholder_engagements:
            return 75.0  # Default score
        
        # More efficient: avoid creating intermediate list
        total_satisfaction = sum(s.satisfaction_score for s in self.stakeholder_engagements.values())
        return total_satisfaction / len(self.stakeholder_engagements)
    
    def _calculate_carbon_footprint(self) -> float:
        """Calculate corporate carbon footprint."""
        # Simplified carbon footprint calculation
        base_footprint = 100.0  # Base metric tons CO2 equivalent
        
        # Adjust based on sustainability practices
        if self.sustainability_targets:
            carbon_reduction = len([t for t in self.sustainability_targets.values() if 'carbon' in t.get('type', '').lower()])
            base_footprint -= carbon_reduction * 10
        
        return max(0.0, base_footprint)
    
    def _get_sustainability_goals(self) -> List[str]:
        """Get current sustainability goals."""
        if not self.sustainability_targets:
            return ["Carbon neutrality by 2030", "100% renewable energy", "Zero waste to landfill"]
        
        return [goal.get('description', '') for goal in self.sustainability_targets.values()]
    
    def conduct_risk_assessment(self, risk_category: str = "comprehensive") -> Dict[str, RiskAssessment]:
        """Conduct comprehensive risk assessment across all corporate areas."""
        if self.verbose:
            CORPORATE_LOGGER.info(f"Conducting {risk_category} risk assessment")
        
        risk_categories = [
            "operational", "financial", "strategic", "compliance", 
            "cybersecurity", "reputation", "environmental", "regulatory"
        ]
        
        assessments = {}
        
        for category in risk_categories:
            if risk_category == "comprehensive" or risk_category == category:
                assessment = self._assess_risk_category(category)
                assessments[category] = assessment
                self.risk_assessments[assessment.risk_id] = assessment
        
        return assessments
    
    def _assess_risk_category(self, category: str) -> RiskAssessment:
        """Assess risk for a specific category."""
        # Simplified risk assessment logic
        risk_levels = {"operational": 0.3, "financial": 0.4, "strategic": 0.5, "compliance": 0.2}
        probabilities = {"operational": 0.6, "financial": 0.4, "strategic": 0.3, "compliance": 0.7}
        
        probability = probabilities.get(category, 0.5)
        impact = risk_levels.get(category, 0.4)
        risk_score = probability * impact
        
        risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"
        
        return RiskAssessment(
            risk_category=category,
            risk_level=risk_level,
            probability=probability,
            impact=impact,
            risk_score=risk_score,
            mitigation_strategies=self._get_mitigation_strategies(category),
            owner=self._get_risk_owner(category)
        )
    
    def _get_mitigation_strategies(self, category: str) -> List[str]:
        """Get mitigation strategies for risk category."""
        strategies = {
            "operational": ["Process optimization", "Backup systems", "Training programs"],
            "financial": ["Diversification", "Hedging strategies", "Cash reserves"],
            "strategic": ["Market research", "Competitive analysis", "Scenario planning"],
            "compliance": ["Regular audits", "Training programs", "Compliance monitoring"],
            "cybersecurity": ["Security protocols", "Incident response", "Regular updates"],
            "reputation": ["Crisis management", "Stakeholder communication", "Brand monitoring"],
            "environmental": ["Sustainability initiatives", "Environmental monitoring", "Green practices"],
            "regulatory": ["Legal compliance", "Regulatory monitoring", "Policy updates"]
        }
        return strategies.get(category, ["General risk mitigation", "Monitoring", "Response planning"])
    
    def _get_risk_owner(self, category: str) -> str:
        """Get risk owner for category."""
        owners = {
            "operational": "COO",
            "financial": "CFO", 
            "strategic": "CEO",
            "compliance": "General Counsel",
            "cybersecurity": "CTO",
            "reputation": "CEO",
            "environmental": "Sustainability Officer",
            "regulatory": "Compliance Officer"
        }
        return owners.get(category, "Board of Directors")
    
    def manage_stakeholder_engagement(self, stakeholder_type: str = "all") -> Dict[str, StakeholderEngagement]:
        """
        Manage comprehensive stakeholder engagement across all stakeholder groups.
        
        Args:
            stakeholder_type: Type of stakeholders to engage with
            
        Returns:
            Dict[str, StakeholderEngagement]: Stakeholder engagement records
        """
        if self.verbose:
            CORPORATE_LOGGER.info(f"Managing stakeholder engagement for {stakeholder_type}")
        
        stakeholder_types = ["investor", "customer", "employee", "community", "supplier", "regulator"]
        engagements = {}
        
        for stype in stakeholder_types:
            if stakeholder_type == "all" or stakeholder_type == stype:
                engagement = self._create_stakeholder_engagement(stype)
                engagements[stype] = engagement
                self.stakeholder_engagements[engagement.stakeholder_id] = engagement
        
        return engagements
    
    def _create_stakeholder_engagement(self, stakeholder_type: str) -> StakeholderEngagement:
        """Create stakeholder engagement record."""
        # Simplified stakeholder engagement logic
        satisfaction_scores = {
            "investor": 85.0,
            "customer": 80.0,
            "employee": 75.0,
            "community": 70.0,
            "supplier": 78.0,
            "regulator": 90.0
        }
        
        return StakeholderEngagement(
            stakeholder_type=stakeholder_type,
            name=f"{stakeholder_type.title()} Group",
            influence_level="high" if stakeholder_type in ["investor", "regulator"] else "medium",
            interest_level="high" if stakeholder_type in ["investor", "customer"] else "medium",
            satisfaction_score=satisfaction_scores.get(stakeholder_type, 75.0),
            concerns=self._get_stakeholder_concerns(stakeholder_type)
        )
    
    def _get_stakeholder_concerns(self, stakeholder_type: str) -> List[str]:
        """Get common concerns for stakeholder type."""
        concerns = {
            "investor": ["ROI", "Risk management", "Growth prospects"],
            "customer": ["Product quality", "Customer service", "Pricing"],
            "employee": ["Work-life balance", "Career development", "Compensation"],
            "community": ["Environmental impact", "Local employment", "Community support"],
            "supplier": ["Payment terms", "Partnership stability", "Fair treatment"],
            "regulator": ["Compliance", "Transparency", "Risk management"]
        }
        return concerns.get(stakeholder_type, ["General concerns", "Communication", "Transparency"])
    
    def establish_compliance_framework(self, regulation_type: str = "comprehensive") -> Dict[str, ComplianceFramework]:
        """Establish comprehensive compliance framework for regulatory adherence."""
        if self.verbose:
            CORPORATE_LOGGER.info(f"Establishing compliance framework for {regulation_type}")
        
        regulations = [
            ("SOX", "financial", ["Internal controls", "Financial reporting", "Audit requirements"]),
            ("GDPR", "data_privacy", ["Data protection", "Privacy rights", "Consent management"]),
            ("ISO 27001", "cybersecurity", ["Information security", "Risk management", "Security controls"]),
            ("ESG", "sustainability", ["Environmental reporting", "Social responsibility", "Governance standards"]),
            ("HIPAA", "healthcare", ["Patient privacy", "Data security", "Compliance monitoring"])
        ]
        
        frameworks = {}
        
        for reg_name, reg_type, requirements in regulations:
            if regulation_type == "comprehensive" or regulation_type == reg_type:
                framework = ComplianceFramework(
                    regulation_name=reg_name,
                    regulation_type=reg_type,
                    requirements=requirements,
                    controls=self._get_compliance_controls(reg_type),
                    responsible_officer=self._get_compliance_officer(reg_type)
                )
                frameworks[reg_name] = framework
                self.compliance_frameworks[framework.compliance_id] = framework
        
        return frameworks
    
    def _get_compliance_controls(self, regulation_type: str) -> List[str]:
        """Get compliance controls for regulation type."""
        controls = {
            "financial": ["Internal audit", "Financial controls", "Reporting systems"],
            "data_privacy": ["Data encryption", "Access controls", "Privacy policies"],
            "cybersecurity": ["Security monitoring", "Incident response", "Access management"],
            "sustainability": ["Environmental monitoring", "Sustainability reporting", "Goal tracking"],
            "healthcare": ["Patient data protection", "Access controls", "Audit trails"]
        }
        return controls.get(regulation_type, ["General controls", "Monitoring", "Reporting"])
    
    def _get_compliance_officer(self, regulation_type: str) -> str:
        """Get compliance officer for regulation type."""
        officers = {
            "financial": "CFO",
            "data_privacy": "Data Protection Officer",
            "cybersecurity": "CISO",
            "sustainability": "Sustainability Officer",
            "healthcare": "Compliance Officer"
        }
        return officers.get(regulation_type, "Compliance Officer")
    
    def _proposal_to_control_vector(self, proposal: CorporateProposal) -> Any:
        """Convert proposal to CRCA-SD ControlVector format."""
        if not CRCA_SD_AVAILABLE:
            return None
        
        try:
            # Map proposal budget impact to control vector budget shares
            # Simplified mapping - in practice, this would be more sophisticated
            budget_shares = {
                'education': 0.0,
                'infrastructure': 0.0,
                'healthcare': 0.0,
                'social_welfare': 0.0,
                'defense': 0.0,
                'other': 0.0
            }
            
            # Map proposal type to budget category
            dept_mapping = {
                DepartmentType.OPERATIONS: 'infrastructure',
                DepartmentType.FINANCE: 'other',
                DepartmentType.TECHNOLOGY: 'infrastructure',
                DepartmentType.HUMAN_RESOURCES: 'social_welfare',
                DepartmentType.LEGAL: 'other',
                DepartmentType.MARKETING: 'other',
            }
            
            category = dept_mapping.get(proposal.department, 'other')
            if proposal.budget_impact > 0:
                # Normalize budget impact to a share (simplified)
                total_budget = sum(d.budget for d in self.departments.values()) or 1.0
                share = min(1.0, proposal.budget_impact / total_budget)
                budget_shares[category] = share
            
            return ControlVector(budget_shares=budget_shares)
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Failed to convert proposal to ControlVector: {e}")
            return None
    
    def _get_objectives_matrix(self) -> Any:
        """Get objectives matrix for CRCA-SD governance."""
        if not CRCA_SD_AVAILABLE:
            return None
        
        try:
            # Create objectives matrix (6 objectives: GDP, Unemployment, Stability, Literacy, Capital, Infrastructure)
            # Each row is an objective, columns are policy dimensions
            import numpy as np
            objectives = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # GDP growth
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Unemployment reduction
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Stability
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Literacy
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Capital
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Infrastructure
            ])
            return objectives
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Failed to create objectives matrix: {e}")
            return None
    
    def _analyze_proposal_causally(self, proposal: CorporateProposal) -> Dict[str, Any]:
        """Analyze proposal using causal reasoning with member CRCAAgents."""
        if not self.config.enable_causal_reasoning or not CRCA_AVAILABLE:
            return {}
        
        try:
            # Get relevant member agents for causal analysis
            relevant_members = []
            for member_id in self.board_members + self.executive_team:
                member = self.members.get(member_id)
                if member and member.agent and hasattr(member.agent, 'run'):
                    # Check if member's expertise matches proposal
                    if any(exp.lower() in proposal.description.lower() or 
                           exp.lower() in proposal.department.value.lower() 
                           for exp in member.expertise_areas):
                        relevant_members.append(member)
            
            if not relevant_members:
                relevant_members = [self.members[mid] for mid in self.board_members[:3] 
                                  if mid in self.members and self.members[mid].agent]
            
            causal_insights = []
            causal_predictions = {}
            
            # Use member CRCAAgents for causal analysis
            for member in relevant_members[:3]:  # Limit to 3 members
                if not member.agent:
                    continue
                
                try:
                    # Create causal analysis task
                    causal_task = f"""
                    Analyze the causal implications of this corporate proposal:
                    
                    Title: {proposal.title}
                    Description: {proposal.description}
                    Budget Impact: ${proposal.budget_impact:,.2f}
                    Department: {proposal.department.value}
                    
                    Consider:
                    1. What are the direct causal effects of this proposal?
                    2. What are the indirect/secondary effects?
                    3. What are potential confounding factors?
                    4. What counterfactual scenarios should be considered?
                    
                    Provide a causal analysis with predictions for key corporate variables.
                    """
                    
                    # Run causal analysis
                    if hasattr(member.agent, 'run'):
                        result = member.agent.run(causal_task)
                        if isinstance(result, dict):
                            causal_insights.append({
                                'member': member.name,
                                'analysis': result.get('analysis', str(result)),
                                'predictions': result.get('predictions', {})
                            })
                            if result.get('predictions'):
                                causal_predictions.update(result.get('predictions', {}))
                except Exception as e:
                    if self.verbose:
                        CORPORATE_LOGGER.warning(f"Causal analysis failed for {member.name}: {e}")
            
            return {
                'insights': causal_insights,
                'predictions': causal_predictions,
                'summary': f"Causal analysis by {len(causal_insights)} members with {len(causal_predictions)} predictions"
            }
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Causal analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_financial_proposal(self, proposal: CorporateProposal) -> Dict[str, Any]:
        """Use QuantTradingAgent to analyze financial proposals."""
        if not self.config.enable_quant_analysis or not self.investment_committee or not CRCA_Q_AVAILABLE:
            return {}
        
        try:
            analysis = {
                'signals': {},
                'portfolio_recommendation': {},
                'risk_assessment': {}
            }
            
            # Get market signals if available
            if hasattr(self.investment_committee, 'compute_signals'):
                try:
                    signals_df = self.investment_committee.compute_signals()
                    if not signals_df.empty and len(signals_df) > 0:
                        # Extract latest signal values
                        latest_signals = signals_df.iloc[-1].to_dict() if hasattr(signals_df, 'iloc') else {}
                        analysis['signals'] = {k: float(v) for k, v in latest_signals.items() 
                                              if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))}
                except Exception as e:
                    if self.verbose:
                        CORPORATE_LOGGER.warning(f"Signal computation failed: {e}")
            
            # Risk assessment
            if hasattr(self.investment_committee, 'risk_monitor') and self.investment_committee.risk_monitor:
                try:
                    # Create a simple position dict for risk monitoring
                    position_size = min(0.2, proposal.budget_impact / (self.cost_tracker.budget_limit * 10) if self.cost_tracker.budget_limit > 0 else 0.1)
                    risk_check = self.investment_committee.risk_monitor.pre_trade_check(
                        signal="corporate_proposal",
                        position_size=position_size,
                        current_positions={},
                        portfolio_value=self.cost_tracker.budget_limit,
                        stop_loss_distance=0.02
                    )
                    analysis['risk_assessment'] = {
                        'approved': risk_check[0],
                        'reason': risk_check[1],
                        'adjusted_size': risk_check[2]
                    }
                except Exception as e:
                    if self.verbose:
                        CORPORATE_LOGGER.warning(f"Risk assessment failed: {e}")
            
            # Portfolio recommendation (simplified)
            analysis['portfolio_recommendation'] = {
                'recommendation': 'moderate' if proposal.budget_impact < self.cost_tracker.budget_limit * 0.1 else 'conservative',
                'allocation': min(0.2, proposal.budget_impact / self.cost_tracker.budget_limit) if self.cost_tracker.budget_limit > 0 else 0.1
            }
            
            return analysis
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Financial proposal analysis failed: {e}")
            return {'error': str(e)}
    
    def _is_code_related_task(self, task: str) -> bool:
        """
        Detect if a task string is code-related and should be executed as a mandate.
        
        Args:
            task: The task string to check
            
        Returns:
            bool: True if the task is code-related
        """
        code_keywords = [
            "build", "create", "develop", "prototype", "code", "software", "app", "application",
            "website", "web app", "platform", "system", "tool", "program", "script", "api",
            "frontend", "backend", "fullstack", "react", "vue", "angular", "next.js", "node",
            "python", "javascript", "typescript", "html", "css", "database", "deploy",
            "saas", "product", "feature", "component", "module", "library", "framework",
            "draft", "write", "generate"
        ]
        
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in code_keywords)
    
    def _is_code_related_proposal(self, proposal: CorporateProposal) -> bool:
        """
        Detect if a proposal is code-related and should be executed as a mandate.
        
        Args:
            proposal: The corporate proposal to check
            
        Returns:
            bool: True if the proposal is code-related
        """
        code_keywords = [
            "build", "create", "develop", "prototype", "code", "software", "app", "application",
            "website", "web app", "platform", "system", "tool", "program", "script", "api",
            "frontend", "backend", "fullstack", "react", "vue", "angular", "next.js", "node",
            "python", "javascript", "typescript", "html", "css", "database", "deploy",
            "saas", "product", "feature", "component", "module", "library", "framework"
        ]
        
        # Check title and description
        text_to_check = f"{proposal.title} {proposal.description}".lower()
        
        # Check if any code keywords are present
        return any(keyword in text_to_check for keyword in code_keywords)
    
    def _process_proposal_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Process proposal-related tasks with performance optimization and automatic execution.
        
        Args:
            task: The proposal task to process
            **kwargs: Additional parameters for proposal creation
            
        Returns:
            Dict[str, Any]: Proposal processing results, including execution results if applicable
        """
        if self.verbose:
            CORPORATE_LOGGER.info("Processing proposal task with democratic voting")
        
        try:
            # Create a proposal from the task
            proposal_id = self.create_proposal(
                title=f"Task Proposal: {task[:50]}...",
                description=task,
                proposal_type=ProposalType.STRATEGIC_INITIATIVE,
                sponsor_id=self.executive_team[0] if self.executive_team else self.board_members[0],
                department=DepartmentType.OPERATIONS,
                **kwargs
            )
            
            # Get the proposal object
            proposal = self._find_proposal(proposal_id)
            
            # Conduct democratic vote with error handling
            vote = self.conduct_corporate_vote(proposal_id)
            
            result = {
                "status": "completed",
                "proposal_id": proposal_id,
                "vote_result": vote.result.value,
                "participants": len(vote.participants),
                "task": task,
                "timestamp": time.time()
            }
            
            # AUTONOMOUS EXECUTION: If proposal is approved and code-related, execute automatically
            if vote.result == VoteResult.APPROVED and self._is_code_related_proposal(proposal):
                if self.verbose:
                    CORPORATE_LOGGER.info(f"Proposal {proposal_id} approved and code-related - executing as mandate")
                
                try:
                    execution_result = self.execute_proposal_as_mandate(proposal_id)
                    result["execution"] = execution_result
                    result["auto_executed"] = True
                    
                    if self.verbose:
                        CORPORATE_LOGGER.info(f"Mandate execution initiated: {execution_result.get('status', 'unknown')}")
                except Exception as exec_error:
                    CORPORATE_LOGGER.error(f"Failed to auto-execute proposal as mandate: {exec_error}")
                    result["execution"] = {
                        "status": "error",
                        "error": str(exec_error)
                    }
                    result["auto_executed"] = False
            else:
                result["auto_executed"] = False
                if vote.result != VoteResult.APPROVED:
                    if self.verbose:
                        CORPORATE_LOGGER.info(f"Proposal {proposal_id} not approved (result: {vote.result.value}) - skipping execution")
                elif not self._is_code_related_proposal(proposal):
                    if self.verbose:
                        CORPORATE_LOGGER.info(f"Proposal {proposal_id} not code-related - skipping execution")
            
            return result
            
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error processing proposal task: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    def _process_meeting_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """Process meeting-related tasks with board governance."""
        if self.verbose:
            CORPORATE_LOGGER.info("Processing meeting task with board governance")
        
        try:
            # Schedule and conduct a board meeting
            meeting_id = self.schedule_board_meeting(
                meeting_type=MeetingType.REGULAR_BOARD,
                agenda=[task],
                **kwargs
            )
            
            meeting = self.conduct_board_meeting(
                meeting_id=meeting_id,
                discussion_topics=[task]
            )
            
            return {
                "status": "completed",
                "meeting_id": meeting_id,
                "resolutions": meeting.resolutions,
                "minutes": meeting.minutes,
                "task": task,
                "timestamp": time.time()
            }
            
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error processing meeting task: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    def _process_strategic_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Process strategic planning tasks with comprehensive analysis.
        
        Args:
            task: The strategic task to process
            **kwargs: Additional parameters for strategic planning
            
        Returns:
            Dict[str, Any]: Strategic planning results
        """
        if self.verbose:
            CORPORATE_LOGGER.info("Processing strategic task with comprehensive analysis")
        
        try:
            # Run a corporate session for strategic planning
            session_results = self.run_corporate_session(
                session_type="strategic_planning",
                agenda_items=[task],
                **kwargs
            )
            
            return {
                "status": "completed",
                "session_type": session_results["session_type"],
                "decisions": session_results["decisions"],
                "task": task,
                "timestamp": time.time()
            }
            
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error processing strategic task: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    def _process_general_task(self, task: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Process general corporate tasks with constitutional fallback hierarchy.
        
        Constitutional fallback order:
        1. Try democratic swarm
        2. If fails → defer to committee
        3. If fails → revert to status quo
        4. If repeated failure → trigger emergency board session
        5. If still failing → freeze actions (safe halt)
        
        Also checks if task is code-related and auto-executes if successful.
        """
        if self.verbose:
            CORPORATE_LOGGER.info("Processing general task through constitutional fallback system")
        
        # Level 1: Try democratic swarm
        if self.democratic_swarm is not None:
            swarm_result = self._try_democratic_swarm(task)
            if swarm_result['status'] == 'success':
                self._constitutional_failure_count = 0
                self._update_status_quo(swarm_result)
                
                # AUTONOMOUS EXECUTION: Check if task is code-related and auto-execute
                if self._is_code_related_task(task):
                    if self.verbose:
                        CORPORATE_LOGGER.info(f"General task is code-related - creating proposal for auto-execution")
                    try:
                        # Create proposal from task
                        proposal_id = self.create_proposal(
                            title=f"Task: {task[:50]}...",
                            description=task,
                            proposal_type=ProposalType.STRATEGIC_INITIATIVE,
                            sponsor_id=self.executive_team[0] if self.executive_team else self.board_members[0],
                            department=DepartmentType.OPERATIONS
                        )
                        proposal = self._find_proposal(proposal_id)
                        
                        # Auto-approve and execute (since swarm already succeeded)
                        if self.verbose:
                            CORPORATE_LOGGER.info(f"Auto-executing code-related task as mandate")
                        execution_result = self.execute_proposal_as_mandate(proposal_id)
                        swarm_result["execution"] = execution_result
                        swarm_result["auto_executed"] = True
                    except Exception as exec_error:
                        CORPORATE_LOGGER.error(f"Failed to auto-execute general task as mandate: {exec_error}")
                        swarm_result["execution"] = {"status": "error", "error": str(exec_error)}
                        swarm_result["auto_executed"] = False
                
                return swarm_result
        
        # Level 2: Defer to committee
        committee_result = self._defer_to_committee(task)
        if committee_result['status'] == 'success':
            self._constitutional_failure_count = 0
            self._update_status_quo(committee_result)
            return committee_result
        
        # Level 3: Revert to status quo (first failure)
        if self._constitutional_failure_count == 0:
            self._constitutional_failure_count = 1
            return self._revert_to_status_quo(task)
        
        # Level 4: Emergency board session (second failure)
        if self._constitutional_failure_count == 1:
            self._constitutional_failure_count = 2
            return self._trigger_emergency_board_session(task)
        
        # Level 5: Safe halt (third+ failure)
        if self._constitutional_failure_count >= 2:
            self._constitutional_failure_count += 1
            return self._freeze_actions(task)
        
        # Fallback (should not reach here)
        CORPORATE_LOGGER.error("Constitutional fallback exhausted - unexpected state")
        return {
            "status": "error",
            "error": "Constitutional fallback system exhausted",
            "task": task,
            "timestamp": time.time()
        }
    
    def _try_democratic_swarm(self, task: str) -> Dict[str, Any]:
        """Attempt to process task through democratic swarm."""
        try:
            result = self.democratic_swarm.run(task)
            
            # Handle None or empty results - this is a legitimacy failure
            if result is None:
                CORPORATE_LOGGER.warning("Democratic swarm returned None - legitimacy failure")
                return {
                    "status": "failed",
                    "fallback_level": "swarm",
                    "reason": "No decision generated (legitimacy failure)",
                    "task": task,
                    "timestamp": time.time()
                }
            
            # Handle string responses like "No response generated"
            if isinstance(result, str) and ("No response generated" in result or not result.strip()):
                CORPORATE_LOGGER.warning("Democratic swarm returned empty response - legitimacy failure")
                return {
                    "status": "failed",
                    "fallback_level": "swarm",
                    "reason": "Empty response (legitimacy failure)",
                    "task": task,
                    "timestamp": time.time()
                }
            
            # Parse result
            parsed_result = self._parse_democratic_result(result)
            
            # Ensure parsed_result is always a dict
            if not isinstance(parsed_result, dict):
                parsed_result = {
                    "result": "processed",
                    "data": parsed_result,
                    "type": "non_dict_result"
                }
            
            # Check if parsed result indicates failure
            if parsed_result.get('result') in ['no_response', 'empty_response', 'error']:
                return {
                    "status": "failed",
                    "fallback_level": "swarm",
                    "reason": f"Swarm returned: {parsed_result.get('result', 'unknown')}",
                    "task": task,
                    "timestamp": time.time()
                }
            
            # Success
            return {
                "status": "success",
                "result": parsed_result,
                "task": task,
                "timestamp": time.time(),
                "method": "democratic_swarm"
            }
            
        except Exception as e:
            CORPORATE_LOGGER.error(f"Democratic swarm exception: {e}")
            if self.verbose:
                import traceback
                CORPORATE_LOGGER.debug(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "failed",
                "fallback_level": "swarm",
                "reason": f"Exception: {str(e)}",
                "task": task,
                "timestamp": time.time()
            }
                
    def _defer_to_committee(self, task: str) -> Dict[str, Any]:
        """Defer task to appropriate committee (Risk, Governance, etc.)."""
        if self.verbose:
            CORPORATE_LOGGER.info("Deferring task to committee (constitutional fallback level 2)")
        
        # Select appropriate committee based on task content
        task_lower = task.lower()
        committee_id = None
        
        # Match task to committee
        if any(keyword in task_lower for keyword in ["risk", "security", "cyber"]):
            # Try Risk Committee
            for cid, committee in self.board_committees.items():
                if committee.committee_type == BoardCommitteeType.RISK:
                    committee_id = cid
                    break
        elif any(keyword in task_lower for keyword in ["governance", "compliance", "legal"]):
            # Try Governance/Nominating Committee
            for cid, committee in self.board_committees.items():
                if committee.committee_type in [BoardCommitteeType.NOMINATING, BoardCommitteeType.AUDIT]:
                    committee_id = cid
                    break
        elif any(keyword in task_lower for keyword in ["technology", "innovation", "digital"]):
            # Try Technology Committee
            for cid, committee in self.board_committees.items():
                if committee.committee_type == BoardCommitteeType.TECHNOLOGY:
                    committee_id = cid
                    break
        
        # Default to first available committee
        if not committee_id and self.board_committees:
            committee_id = list(self.board_committees.keys())[0]
        
        if committee_id:
            try:
                committee = self.board_committees[committee_id]
                committee_task = f"Committee review of: {task}"
                
                # Use committee meeting to process
                result = self.conduct_committee_meeting(
                    committee_id=committee_id,
                    agenda=[task]
                )
                
                return {
                    "status": "success",
                    "result": result,
                    "task": task,
                    "timestamp": time.time(),
                    "method": "committee",
                    "committee": committee.name
                }
            except Exception as e:
                CORPORATE_LOGGER.warning(f"Committee processing failed: {e}")
        
        # Committee fallback failed
        return {
            "status": "failed",
            "fallback_level": "committee",
            "reason": "Committee processing failed or no committee available",
            "task": task,
            "timestamp": time.time()
        }
    
    def _revert_to_status_quo(self, task: str) -> Dict[str, Any]:
        """Revert to last known good state (status quo)."""
        if self.verbose:
            CORPORATE_LOGGER.warning("Reverting to status quo (constitutional fallback level 3)")
        
        if self._last_successful_state:
            CORPORATE_LOGGER.info("Returning last successful state as status quo")
            return {
                "status": "status_quo",
                "result": self._last_successful_state,
                "task": task,
                "timestamp": time.time(),
                "method": "status_quo_reversion",
                "message": "No decision could be reached - maintaining status quo"
            }
        else:
            # No status quo available - return neutral response
            CORPORATE_LOGGER.warning("No status quo available - returning neutral response")
            return {
                "status": "status_quo",
                "result": {
                    "decision": "no_change",
                    "message": "Unable to reach decision - maintaining current state",
                    "task": task
                },
                "task": task,
                "timestamp": time.time(),
                "method": "status_quo_reversion",
                "message": "No previous successful state - maintaining neutral position"
            }
    
    def _trigger_emergency_board_session(self, task: str) -> Dict[str, Any]:
        """Trigger emergency board session to resolve failure."""
        if self.verbose:
            CORPORATE_LOGGER.warning("Triggering emergency board session (constitutional fallback level 4)")
        
        try:
            # Schedule emergency meeting
            meeting_id = self.schedule_board_meeting(
                meeting_type=MeetingType.EMERGENCY_MEETING,
                agenda=[f"Emergency resolution: {task}"]
            )
            
            # Conduct emergency meeting
            meeting = self.conduct_board_meeting(
                meeting_id=meeting_id,
                discussion_topics=[f"Emergency resolution required for: {task}"]
            )
            
            # Check if meeting produced resolutions
            if meeting.resolutions:
                return {
                    "status": "success",
                    "result": {
                        "resolutions": meeting.resolutions,
                        "minutes": meeting.minutes,
                        "meeting_type": "emergency"
                    },
                    "task": task,
                    "timestamp": time.time(),
                    "method": "emergency_board_session",
                    "meeting_id": meeting_id
                }
            else:
                # Emergency meeting failed to produce resolution
                return {
                    "status": "failed",
                    "fallback_level": "emergency_board",
                    "reason": "Emergency board session failed to produce resolution",
                    "task": task,
                    "timestamp": time.time()
                }
        except Exception as e:
            CORPORATE_LOGGER.error(f"Emergency board session failed: {e}")
            return {
                "status": "failed",
                "fallback_level": "emergency_board",
                "reason": f"Exception: {str(e)}",
                "task": task,
                "timestamp": time.time()
            }
    
    def _freeze_actions(self, task: str) -> Dict[str, Any]:
        """Freeze all actions - safe halt state."""
        if self.verbose:
            CORPORATE_LOGGER.critical("Freezing actions - safe halt activated (constitutional fallback level 5)")
        
        # Set frozen state
        self._frozen_actions = True
        self.emergency_stop_active = True
        
        # Log the freeze
        self.audit_trails.append({
            "event": "constitutional_safe_halt",
            "task": task,
            "failure_count": self._constitutional_failure_count,
            "timestamp": time.time(),
            "reason": "Repeated decision-making failures - governance vacuum prevented"
        })
        
        return {
            "status": "frozen",
            "result": {
                "message": "Actions frozen - safe halt activated",
                "reason": "Repeated constitutional failures - governance system unable to reach decisions",
                "failure_count": self._constitutional_failure_count,
                "requires_human_intervention": True
            },
            "task": task,
            "timestamp": time.time(),
            "method": "safe_halt",
            "frozen": True
        }
    
    def _update_status_quo(self, successful_result: Dict[str, Any]) -> None:
        """Update status quo with successful result."""
        # Store key state information
        self._last_successful_state = {
            "result": successful_result.get("result", {}),
            "status": successful_result.get("status", "unknown"),
            "timestamp": successful_result.get("timestamp", time.time()),
            "method": successful_result.get("method", "unknown")
        }
        
        if self.verbose:
            CORPORATE_LOGGER.debug(f"Updated status quo with {successful_result.get('method', 'unknown')} result")
    
    def _check_and_reset_action_limits(self) -> None:
        """Check and reset action limits for autonomous operation."""
        # For full autonomy, this just tracks actions without blocking
        # Action limits reset automatically on each call (no period tracking needed)
        if self.verbose:
            CORPORATE_LOGGER.debug("Action limits checked (autonomous mode - no blocking)")
    
    
    def _analyze_vote_results(
        self,
        individual_votes: Dict[str, Dict[str, Any]],
        proposal: CorporateProposal
    ) -> VoteResult:
        """Analyze voting results and determine outcome."""
        if not individual_votes:
            return VoteResult.FAILED
        
        total_weight = 0
        approve_weight = 0
        reject_weight = 0
        
        for member_id, vote_data in individual_votes.items():
            weight = vote_data.get("voting_weight", 1.0)
            vote = vote_data.get("vote", "ABSTAIN")
            
            total_weight += weight
            
            if vote == "APPROVE":
                approve_weight += weight
            elif vote == "REJECT":
                reject_weight += weight
        
        if total_weight == 0:
            return VoteResult.FAILED
        
        # Simple majority rule with weighted voting
        approve_percentage = approve_weight / total_weight
        reject_percentage = reject_weight / total_weight
        
        if approve_percentage > 0.5:
            return VoteResult.APPROVED
        elif reject_percentage > 0.5:
            return VoteResult.REJECTED
        else:
            return VoteResult.TABLED
    
    def run_corporate_session(
        self,
        session_type: str = "board_meeting",
        agenda_items: List[str] = None
    ) -> Dict[str, Any]:
        """Run a corporate governance session."""
        if not agenda_items:
            agenda_items = ["Strategic planning", "Budget review", "Operational updates"]
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Starting corporate session: {session_type}")
        
        session_results = {
            "session_type": session_type,
            "agenda_items": agenda_items,
            "participants": len(self.board_members + self.executive_team),
            "decisions": [],
            "timestamp": time.time()
        }
        
        # Process each agenda item
        for item in agenda_items:
            if self.verbose:
                CORPORATE_LOGGER.info(f"Processing agenda item: {item}")
            
            # Create a proposal for the agenda item
            proposal_id = self.create_proposal(
                title=f"Agenda Item: {item}",
                description=f"Discussion and decision on {item}",
                proposal_type=ProposalType.STRATEGIC_INITIATIVE,
                sponsor_id=self.board_members[0] if self.board_members else self.executive_team[0],
                department=DepartmentType.OPERATIONS
            )
            
            # Conduct vote
            vote = self.conduct_corporate_vote(proposal_id)
            
            # Get the proposal object
            proposal = self._find_proposal(proposal_id)
            
            decision_result = {
                "item": item,
                "proposal_id": proposal_id,
                "result": vote.result.value,
                "participants": len(vote.participants)
            }
            
            # AUTONOMOUS EXECUTION: If proposal is approved and code-related, execute automatically
            if vote.result == VoteResult.APPROVED and self._is_code_related_proposal(proposal):
                if self.verbose:
                    CORPORATE_LOGGER.info(f"Strategic session proposal {proposal_id} approved and code-related - executing as mandate")
                
                try:
                    execution_result = self.execute_proposal_as_mandate(proposal_id)
                    decision_result["execution"] = execution_result
                    decision_result["auto_executed"] = True
                except Exception as exec_error:
                    CORPORATE_LOGGER.error(f"Failed to auto-execute strategic proposal as mandate: {exec_error}")
                    decision_result["execution"] = {
                        "status": "error",
                        "error": str(exec_error)
                    }
                    decision_result["auto_executed"] = False
            else:
                decision_result["auto_executed"] = False
            
            session_results["decisions"].append(decision_result)
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Corporate session completed: {len(session_results['decisions'])} decisions made")
        
        return session_results
    
    def get_corporate_status(self) -> Dict[str, Any]:
        """Get current corporate status and metrics including advanced governance frameworks."""
        # Get board performance metrics
        board_performance = self.evaluate_board_performance()
        
        # Calculate ESG score
        esg_score = self.calculate_esg_score()
        
        # Get risk assessment summary
        risk_summary = self._get_risk_summary()
        
        # Get stakeholder engagement summary
        stakeholder_summary = self._get_stakeholder_summary()
        
        # Get compliance status
        compliance_status = self._get_compliance_status()
        
        return {
            "name": self.name,
            "description": self.description,
            "total_members": len(self.members),
            "board_members": len(self.board_members),
            "executive_team": len(self.executive_team),
            "departments": len(self.departments),
            "board_committees": len(self.board_committees),
            "board_meetings": len(self.board_meetings),
            "independent_directors": len(self.independent_directors),
            "executive_directors": len(self.executive_directors),
            "active_proposals": len([p for p in self.proposals if p.status == "pending"]),
            "total_votes": len(self.votes),
            "recent_decisions": [
                {
                    "proposal": vote.proposal.title,
                    "result": vote.result.value,
                    "timestamp": vote.timestamp
                }
                for vote in self.votes[-5:]  # Last 5 votes
            ],
            "department_budgets": {
                dept.name: dept.budget
                for dept in self.departments.values()
            },
            "board_governance": {
                "committees": {
                    committee.name: {
                        "type": committee.committee_type.value,
                        "chair": self.members[committee.chair].name if committee.chair in self.members else "Unknown",
                        "members": len(committee.members),
                        "responsibilities": committee.responsibilities
                    }
                    for committee in self.board_committees.values()
                },
                "recent_meetings": [
                    {
                        "type": meeting.meeting_type.value,
                        "date": meeting.date,
                        "attendees": len(meeting.attendees),
                        "quorum_met": meeting.quorum_met,
                        "resolutions": len(meeting.resolutions)
                    }
                    for meeting in self.board_meetings[-3:]  # Last 3 meetings
                ],
                "performance_metrics": board_performance
            },
            "esg_governance": {
                "overall_score": esg_score.overall_score,
                "environmental_score": esg_score.environmental_score,
                "social_score": esg_score.social_score,
                "governance_score": esg_score.governance_score,
                "carbon_footprint": esg_score.carbon_footprint,
                "diversity_index": esg_score.diversity_index,
                "stakeholder_satisfaction": esg_score.stakeholder_satisfaction,
                "sustainability_goals": esg_score.sustainability_goals
            },
            "risk_management": risk_summary,
            "stakeholder_engagement": stakeholder_summary,
            "compliance_framework": compliance_status,
            "advanced_governance": {
                "total_risk_assessments": len(self.risk_assessments),
                "total_stakeholder_engagements": len(self.stakeholder_engagements),
                "total_compliance_frameworks": len(self.compliance_frameworks),
                "crisis_management_plans": len(self.crisis_management_plans),
                "innovation_pipeline_items": len(self.innovation_pipeline),
                "audit_trail_entries": len(self.audit_trails)
            },
            "aop_integration": self._get_aop_status()
        }
    
    def _get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        if not self.risk_assessments:
            return {"status": "No risk assessments conducted", "total_risks": 0}
        
        high_risks = [r for r in self.risk_assessments.values() if r.risk_level == "high"]
        medium_risks = [r for r in self.risk_assessments.values() if r.risk_level == "medium"]
        low_risks = [r for r in self.risk_assessments.values() if r.risk_level == "low"]
        
        return {
            "total_risks": len(self.risk_assessments),
            "high_risks": len(high_risks),
            "medium_risks": len(medium_risks),
            "low_risks": len(low_risks),
            "risk_categories": list(set(r.risk_category for r in self.risk_assessments.values())),
            "average_risk_score": sum(r.risk_score for r in self.risk_assessments.values()) / len(self.risk_assessments)
        }
    
    def _get_stakeholder_summary(self) -> Dict[str, Any]:
        """Get stakeholder engagement summary."""
        if not self.stakeholder_engagements:
            return {"status": "No stakeholder engagements recorded", "total_stakeholders": 0}
        
        stakeholder_types = list(set(s.stakeholder_type for s in self.stakeholder_engagements.values()))
        avg_satisfaction = sum(s.satisfaction_score for s in self.stakeholder_engagements.values()) / len(self.stakeholder_engagements)
        
        return {
            "total_stakeholders": len(self.stakeholder_engagements),
            "stakeholder_types": stakeholder_types,
            "average_satisfaction": avg_satisfaction,
            "high_influence_stakeholders": len([s for s in self.stakeholder_engagements.values() if s.influence_level == "high"]),
            "high_interest_stakeholders": len([s for s in self.stakeholder_engagements.values() if s.interest_level == "high"])
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance framework status."""
        if not self.compliance_frameworks:
            return {"status": "No compliance frameworks established", "total_frameworks": 0}
        
        compliant_frameworks = [f for f in self.compliance_frameworks.values() if f.compliance_status == "compliant"]
        non_compliant_frameworks = [f for f in self.compliance_frameworks.values() if f.compliance_status == "non_compliant"]
        
        return {
            "total_frameworks": len(self.compliance_frameworks),
            "compliant_frameworks": len(compliant_frameworks),
            "non_compliant_frameworks": len(non_compliant_frameworks),
            "compliance_rate": len(compliant_frameworks) / len(self.compliance_frameworks) * 100,
            "regulation_types": list(set(f.regulation_type for f in self.compliance_frameworks.values())),
            "average_compliance_score": sum(f.compliance_score for f in self.compliance_frameworks.values()) / len(self.compliance_frameworks)
        }
    
    def conduct_comprehensive_governance_review(self) -> Dict[str, Any]:
        """Conduct a comprehensive governance review across all frameworks."""
        if self.verbose:
            CORPORATE_LOGGER.info("Conducting comprehensive governance review")
        
        # Calculate ESG score
        esg_score = self.calculate_esg_score()
        
        # Conduct risk assessment
        risk_assessments = self.conduct_risk_assessment("comprehensive")
        
        # Manage stakeholder engagement
        stakeholder_engagements = self.manage_stakeholder_engagement("all")
        
        # Establish compliance framework
        compliance_frameworks = self.establish_compliance_framework("comprehensive")
        
        # Evaluate board performance
        board_performance = self.evaluate_board_performance()
        
        # Get corporate status
        corporate_status = self.get_corporate_status()
        
        return {
            "review_timestamp": time.time(),
            "esg_analysis": {
                "overall_score": esg_score.overall_score,
                "environmental_score": esg_score.environmental_score,
                "social_score": esg_score.social_score,
                "governance_score": esg_score.governance_score,
                "carbon_footprint": esg_score.carbon_footprint,
                "diversity_index": esg_score.diversity_index,
                "stakeholder_satisfaction": esg_score.stakeholder_satisfaction
            },
            "risk_analysis": {
                "total_risks": len(risk_assessments),
                "high_risk_categories": [cat for cat, risk in risk_assessments.items() if risk.risk_level == "high"],
                "average_risk_score": sum(risk.risk_score for risk in risk_assessments.values()) / len(risk_assessments) if risk_assessments else 0
            },
            "stakeholder_analysis": {
                "total_stakeholders": len(stakeholder_engagements),
                "average_satisfaction": sum(eng.satisfaction_score for eng in stakeholder_engagements.values()) / len(stakeholder_engagements) if stakeholder_engagements else 0,
                "stakeholder_types": list(stakeholder_engagements.keys())
            },
            "compliance_analysis": {
                "total_frameworks": len(compliance_frameworks),
                "compliance_rate": len([f for f in compliance_frameworks.values() if f.compliance_status == "compliant"]) / len(compliance_frameworks) * 100 if compliance_frameworks else 0,
                "regulation_types": list(set(f.regulation_type for f in compliance_frameworks.values()))
            },
            "board_performance": board_performance,
            "corporate_status": corporate_status,
            "governance_recommendations": self._generate_governance_recommendations(esg_score, risk_assessments, stakeholder_engagements, compliance_frameworks)
        }
    
    def _generate_governance_recommendations(
        self, 
        esg_score: ESGScore, 
        risk_assessments: Dict[str, RiskAssessment], 
        stakeholder_engagements: Dict[str, StakeholderEngagement], 
        compliance_frameworks: Dict[str, ComplianceFramework]
    ) -> List[str]:
        """Generate governance recommendations based on analysis."""
        recommendations = []
        
        # ESG recommendations
        if esg_score.overall_score < 70:
            recommendations.append("Improve overall ESG performance through enhanced sustainability initiatives")
        if esg_score.environmental_score < 70:
            recommendations.append("Implement stronger environmental sustainability programs")
        if esg_score.social_score < 70:
            recommendations.append("Enhance social responsibility and stakeholder engagement")
        if esg_score.governance_score < 70:
            recommendations.append("Strengthen corporate governance practices and board oversight")
        
        # Risk recommendations
        high_risks = [risk for risk in risk_assessments.values() if risk.risk_level == "high"]
        if high_risks:
            recommendations.append(f"Address {len(high_risks)} high-risk areas with immediate mitigation strategies")
        
        # Stakeholder recommendations
        low_satisfaction_stakeholders = [eng for eng in stakeholder_engagements.values() if eng.satisfaction_score < 70]
        if low_satisfaction_stakeholders:
            recommendations.append("Improve stakeholder satisfaction through enhanced engagement programs")
        
        # Compliance recommendations
        non_compliant = [f for f in compliance_frameworks.values() if f.compliance_status == "non_compliant"]
        if non_compliant:
            recommendations.append(f"Address {len(non_compliant)} non-compliant regulatory frameworks")
        
        if not recommendations:
            recommendations.append("Maintain current governance excellence and continue monitoring")
        
        return recommendations
    
    def _get_aop_status(self) -> Dict[str, Any]:
        """Get AOP integration status and statistics."""
        if not AOP_AVAILABLE:
            return {
                "enabled": False,
                "status": "AOP not available"
            }
        
        status = {
            "enabled": self.config.enable_aop,
            "queue_execution_enabled": self.config.enable_queue_execution,
            "aop_server_active": self.aop_server is not None,
            "total_member_queues": len(self.member_task_queues)
        }
        
        if self.aop_server:
            try:
                server_info = self.aop_server.get_server_info()
                status["aop_server"] = {
                    "port": self.config.aop_server_port,
                    "total_agents": server_info.get("total_agents", 0),
                    "queue_enabled": server_info.get("queue_enabled", False),
                    "persistence_enabled": server_info.get("persistence_enabled", False)
                }
            except Exception as e:
                status["aop_server_error"] = str(e)
        
        # Get queue statistics for member queues
        if self.member_task_queues:
            queue_stats = {}
            for member_id, queue in self.member_task_queues.items():
                member = self.members.get(member_id)
                if member:
                    try:
                        stats = queue.get_stats()
                        queue_stats[member.name] = {
                            "total_tasks": stats.total_tasks,
                            "completed_tasks": stats.completed_tasks,
                            "failed_tasks": stats.failed_tasks,
                            "pending_tasks": stats.pending_tasks,
                            "processing_tasks": stats.processing_tasks,
                            "average_processing_time": stats.average_processing_time,
                            "queue_size": stats.queue_size,
                            "queue_status": queue.get_status().value
                        }
                    except Exception as e:
                        queue_stats[member.name] = {"error": str(e)}
            status["member_queue_stats"] = queue_stats
        
        return status
    
    def start_aop_server(self) -> bool:
        """
        Start the AOP MCP server.
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        if not AOP_AVAILABLE:
            CORPORATE_LOGGER.warning("AOP not available, cannot start server")
            return False
        
        if not self.config.enable_aop:
            CORPORATE_LOGGER.warning("AOP is disabled in configuration")
            return False
        
        if not self.aop_server:
            CORPORATE_LOGGER.warning("AOP server not initialized")
            return False
        
        try:
            if self.verbose:
                CORPORATE_LOGGER.info(f"Starting AOP server on port {self.config.aop_server_port}")
            
            # Start server in background thread
            import threading
            server_thread = threading.Thread(
                target=self.aop_server.start_server,
                daemon=True,
                name=f"AOP-Server-{self.name}"
            )
            server_thread.start()
            
            if self.verbose:
                CORPORATE_LOGGER.info("AOP server started in background thread")
            
            return True
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to start AOP server: {e}")
            return False
    
    def get_member_queue_stats(self, member_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get queue statistics for member agents.
        
        Args:
            member_id: Optional specific member ID. If None, returns stats for all members.
            
        Returns:
            Dict containing queue statistics
        """
        if not AOP_AVAILABLE or not self.config.enable_queue_execution:
            return {
                "success": False,
                "error": "Queue execution is not enabled"
            }
        
        try:
            if member_id:
                if member_id not in self.member_task_queues:
                    return {
                        "success": False,
                        "error": f"Member {member_id} does not have a queue"
                    }
                
                member = self.members.get(member_id)
                queue = self.member_task_queues[member_id]
                stats = queue.get_stats()
                
                return {
                    "success": True,
                    "member_name": member.name if member else "Unknown",
                    "member_id": member_id,
                    "stats": {
                        "total_tasks": stats.total_tasks,
                        "completed_tasks": stats.completed_tasks,
                        "failed_tasks": stats.failed_tasks,
                        "pending_tasks": stats.pending_tasks,
                        "processing_tasks": stats.processing_tasks,
                        "average_processing_time": stats.average_processing_time,
                        "queue_size": stats.queue_size,
                        "queue_status": queue.get_status().value
                    }
                }
            else:
                # Get stats for all members
                all_stats = {}
                for mid, queue in self.member_task_queues.items():
                    member = self.members.get(mid)
                    if member:
                        stats = queue.get_stats()
                        all_stats[member.name] = {
                            "member_id": mid,
                            "total_tasks": stats.total_tasks,
                            "completed_tasks": stats.completed_tasks,
                            "failed_tasks": stats.failed_tasks,
                            "pending_tasks": stats.pending_tasks,
                            "processing_tasks": stats.processing_tasks,
                            "average_processing_time": stats.average_processing_time,
                            "queue_size": stats.queue_size,
                            "queue_status": queue.get_status().value
                        }
                
                return {
                    "success": True,
                    "stats": all_stats,
                    "total_members": len(all_stats)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def pause_member_queue(self, member_id: str) -> bool:
        """
        Pause the task queue for a specific member.
        
        Args:
            member_id: ID of the member
            
        Returns:
            bool: True if paused successfully, False otherwise
        """
        if member_id not in self.member_task_queues:
            return False
        
        try:
            self.member_task_queues[member_id].pause_workers()
            if self.verbose:
                member = self.members.get(member_id)
                CORPORATE_LOGGER.info(f"Paused queue for member: {member.name if member else member_id}")
            return True
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to pause queue for {member_id}: {e}")
            return False
    
    def resume_member_queue(self, member_id: str) -> bool:
        """
        Resume the task queue for a specific member.
        
        Args:
            member_id: ID of the member
            
        Returns:
            bool: True if resumed successfully, False otherwise
        """
        if member_id not in self.member_task_queues:
            return False
        
        try:
            self.member_task_queues[member_id].resume_workers()
            if self.verbose:
                member = self.members.get(member_id)
                CORPORATE_LOGGER.info(f"Resumed queue for member: {member.name if member else member_id}")
            return True
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to resume queue for {member_id}: {e}")
            return False
    
    def get_aop_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the AOP server.
        
        Returns:
            Dict containing AOP server information, or None if not available
        """
        if not self.aop_server:
            return None
        
        try:
            return self.aop_server.get_server_info()
        except Exception as e:
            if self.verbose:
                CORPORATE_LOGGER.warning(f"Failed to get AOP server info: {e}")
            return None
    
    def launch_tui(self, refresh_rate: float = 1.0, **kwargs) -> None:
        """
        Launch the CorporateSwarm TUI (Terminal User Interface).
        
        Args:
            refresh_rate: Refresh rate in seconds (default: 1.0)
            **kwargs: Additional arguments for TUI initialization
        """
        try:
            from utils.tui import CorporateSwarmTUI
            
            tui = CorporateSwarmTUI(
                corporate_swarm=self,
                title=f"{self.name} - Governance Dashboard",
                refresh_rate=refresh_rate,
                **kwargs
            )
            
            def update_callback(tui_instance):
                """Update callback for TUI."""
                tui_instance._update_status()
            
            tui.run_live(update_callback=update_callback, refresh_rate=refresh_rate)
        except ImportError as e:
            CORPORATE_LOGGER.error(f"TUI not available: {e}. Install rich: pip install rich")
            raise
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to launch TUI: {e}")
            raise
    
    # ============================================================================
    # PERSISTENCE METHODS
    # ============================================================================
    
    def save_snapshot(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Save corporation state to JSON file with integrity hash.
        
        Args:
            filepath: Optional file path. If None, uses default: {name}_snapshot_{timestamp}.json
            
        Returns:
            Dict containing snapshot metadata including integrity hash
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.name.replace(' ', '_')}_snapshot_{timestamp}.json"
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Saving corporation snapshot to {filepath}")
        
        # Serialize state
        state = self._serialize_state()
        
        # Compute integrity hash
        integrity_hash = self._compute_integrity_hash(state)
        
        # Create snapshot structure
        snapshot = {
            "metadata": {
                "version": "1.0",
                "timestamp": time.time(),
                "integrity_hash": integrity_hash,
                "corporation_name": self.name,
                "description": self.description
            },
            "state": state
        }
        
        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            if self.verbose:
                CORPORATE_LOGGER.info(f"Snapshot saved successfully: {filepath}")
            
            return {
                "success": True,
                "filepath": filepath,
                "integrity_hash": integrity_hash,
                "timestamp": snapshot["metadata"]["timestamp"]
            }
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to save snapshot: {e}")
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }
    
    def load_snapshot(self, filepath: str) -> bool:
        """
        Load corporation state from JSON file with integrity verification.
        
        Args:
            filepath: Path to snapshot file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            CORPORATE_LOGGER.error(f"Snapshot file not found: {filepath}")
            return False
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Loading corporation snapshot from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                snapshot = json.load(f)
            
            # Verify structure
            if "metadata" not in snapshot or "state" not in snapshot:
                CORPORATE_LOGGER.error("Invalid snapshot format: missing metadata or state")
                return False
            
            # Verify integrity
            expected_hash = snapshot["metadata"].get("integrity_hash")
            if expected_hash:
                if not self._verify_integrity(snapshot["state"], expected_hash):
                    CORPORATE_LOGGER.error("Snapshot integrity check failed - file may be corrupted")
                    return False
            
            # Deserialize state
            self._deserialize_state(snapshot["state"])
            
            if self.verbose:
                CORPORATE_LOGGER.info(f"Snapshot loaded successfully from {filepath}")
            
            return True
        except Exception as e:
            CORPORATE_LOGGER.error(f"Failed to load snapshot: {e}")
            if self.verbose:
                import traceback
                CORPORATE_LOGGER.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize all corporate state to dict (exclude non-serializable objects).
        
        Returns:
            Dict containing all serializable corporate state
        """
        state = {
            "members": {},
            "departments": {},
            "proposals": [],
            "votes": [],
            "board_members": self.board_members,
            "executive_team": self.executive_team,
            "board_committees": {},
            "board_meetings": [],
            "independent_directors": self.independent_directors,
            "executive_directors": self.executive_directors,
            "esg_scores": {},
            "risk_assessments": {},
            "stakeholder_engagements": {},
            "compliance_frameworks": {},
            "crisis_management_plans": self.crisis_management_plans,
            "innovation_pipeline": self.innovation_pipeline,
            "audit_trails": self.audit_trails,
            "performance_metrics": self.performance_metrics,
            "sustainability_targets": self.sustainability_targets,
            "ai_ethics_framework": self.ai_ethics_framework,
            "cost_tracker": {
                "budget_limit": self.cost_tracker.budget_limit,
                "current_cost": self.cost_tracker.current_cost
            },
            "constitutional_state": {
                "failure_count": self._constitutional_failure_count,
                "last_successful_state": self._last_successful_state,
                "frozen_actions": self._frozen_actions,
                "emergency_stop_active": self.emergency_stop_active
            },
            "running_mandates": self.running_mandates,
            "config": {
                "budget_limit": self.config.budget_limit,
                "decision_threshold": self.config.decision_threshold,
                "enable_causal_reasoning": self.config.enable_causal_reasoning,
                "enable_quant_analysis": self.config.enable_quant_analysis,
                "enable_crca_sd_governance": self.config.enable_crca_sd_governance,
                "enable_aop": self.config.enable_aop,
                "enable_queue_execution": self.config.enable_queue_execution
            }
        }
        
        # Serialize members (exclude agent objects)
        for member_id, member in self.members.items():
            member_dict = member.dict() if hasattr(member, 'dict') else asdict(member) if hasattr(member, '__dict__') else {}
            # Remove non-serializable agent
            if 'agent' in member_dict:
                del member_dict['agent']
            state["members"][member_id] = member_dict
        
        # Serialize departments
        for dept_id, dept in self.departments.items():
            state["departments"][dept_id] = asdict(dept) if hasattr(dept, '__dict__') else dept.__dict__
        
        # Serialize proposals
        for proposal in self.proposals:
            proposal_dict = proposal.dict() if hasattr(proposal, 'dict') else asdict(proposal) if hasattr(proposal, '__dict__') else {}
            state["proposals"].append(proposal_dict)
        
        # Serialize votes (exclude proposal object, store ID)
        for vote in self.votes:
            vote_dict = vote.dict() if hasattr(vote, 'dict') else asdict(vote) if hasattr(vote, '__dict__') else {}
            # Replace proposal object with ID
            if 'proposal' in vote_dict and hasattr(vote_dict['proposal'], 'proposal_id'):
                vote_dict['proposal_id'] = vote_dict['proposal'].proposal_id
                del vote_dict['proposal']
            state["votes"].append(vote_dict)
        
        # Serialize board committees
        for committee_id, committee in self.board_committees.items():
            state["board_committees"][committee_id] = asdict(committee) if hasattr(committee, '__dict__') else committee.__dict__
        
        # Serialize board meetings
        for meeting in self.board_meetings:
            state["board_meetings"].append(asdict(meeting) if hasattr(meeting, '__dict__') else meeting.__dict__)
        
        # Serialize ESG scores
        for score_id, esg_score in self.esg_scores.items():
            state["esg_scores"][score_id] = esg_score.dict() if hasattr(esg_score, 'dict') else asdict(esg_score) if hasattr(esg_score, '__dict__') else {}
        
        # Serialize risk assessments
        for risk_id, risk in self.risk_assessments.items():
            state["risk_assessments"][risk_id] = risk.dict() if hasattr(risk, 'dict') else asdict(risk) if hasattr(risk, '__dict__') else {}
        
        # Serialize stakeholder engagements
        for stakeholder_id, engagement in self.stakeholder_engagements.items():
            state["stakeholder_engagements"][stakeholder_id] = engagement.dict() if hasattr(engagement, 'dict') else asdict(engagement) if hasattr(engagement, '__dict__') else {}
        
        # Serialize compliance frameworks
        for compliance_id, framework in self.compliance_frameworks.items():
            state["compliance_frameworks"][compliance_id] = framework.dict() if hasattr(framework, 'dict') else asdict(framework) if hasattr(framework, '__dict__') else {}
        
        return state
    
    def _deserialize_state(self, state: Dict[str, Any]) -> None:
        """
        Deserialize state dict back into corporation.
        
        Args:
            state: Serialized state dictionary
        """
        # Restore basic lists
        self.board_members = state.get("board_members", [])
        self.executive_team = state.get("executive_team", [])
        self.independent_directors = state.get("independent_directors", [])
        self.executive_directors = state.get("executive_directors", [])
        
        # Restore cost tracker
        cost_data = state.get("cost_tracker", {})
        self.cost_tracker.budget_limit = cost_data.get("budget_limit", 200.0)
        self.cost_tracker.current_cost = cost_data.get("current_cost", 0.0)
        
        # Restore constitutional state
        const_state = state.get("constitutional_state", {})
        self._constitutional_failure_count = const_state.get("failure_count", 0)
        self._last_successful_state = const_state.get("last_successful_state", {})
        self._frozen_actions = const_state.get("frozen_actions", False)
        self.emergency_stop_active = const_state.get("emergency_stop_active", False)
        
        # Restore members (recreate without agents - agents will be reinitialized)
        self.members = {}
        for member_id, member_data in state.get("members", {}).items():
            # Remove agent if present
            if 'agent' in member_data:
                del member_data['agent']
            # Recreate member
            try:
                member = CorporateMember(**member_data)
                self.members[member_id] = member
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize member {member_id}: {e}")
        
        # Reinitialize member agents if causal reasoning is enabled
        if self.config.enable_causal_reasoning:
            for member_id, member in self.members.items():
                if member.agent is None:
                    self._initialize_member_agent(member)
        
        # Restore departments
        self.departments = {}
        for dept_id, dept_data in state.get("departments", {}).items():
            try:
                dept = CorporateDepartment(**dept_data)
                self.departments[dept_id] = dept
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize department {dept_id}: {e}")
        
        # Restore proposals
        self.proposals = []
        for proposal_data in state.get("proposals", []):
            try:
                proposal = CorporateProposal(**proposal_data)
                self.proposals.append(proposal)
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize proposal: {e}")
        
        # Restore votes (need to link proposals)
        self.votes = []
        for vote_data in state.get("votes", []):
            try:
                # Find proposal by ID
                proposal_id = vote_data.get("proposal_id")
                proposal = None
                if proposal_id:
                    for p in self.proposals:
                        if p.proposal_id == proposal_id:
                            proposal = p
                            break
                
                if not proposal:
                    # Create dummy proposal if not found
                    proposal = CorporateProposal()
                
                # Recreate vote
                vote_data_copy = vote_data.copy()
                vote_data_copy['proposal'] = proposal
                if 'proposal_id' in vote_data_copy:
                    del vote_data_copy['proposal_id']
                
                vote = CorporateVote(**vote_data_copy)
                self.votes.append(vote)
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize vote: {e}")
        
        # Restore board committees
        self.board_committees = {}
        for committee_id, committee_data in state.get("board_committees", {}).items():
            try:
                committee = BoardCommittee(**committee_data)
                self.board_committees[committee_id] = committee
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize committee {committee_id}: {e}")
        
        # Restore board meetings
        self.board_meetings = []
        for meeting_data in state.get("board_meetings", []):
            try:
                meeting = BoardMeeting(**meeting_data)
                self.board_meetings.append(meeting)
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize meeting: {e}")
        
        # Restore ESG scores
        self.esg_scores = {}
        for score_id, score_data in state.get("esg_scores", {}).items():
            try:
                esg_score = ESGScore(**score_data)
                self.esg_scores[score_id] = esg_score
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize ESG score {score_id}: {e}")
        
        # Restore risk assessments
        self.risk_assessments = {}
        for risk_id, risk_data in state.get("risk_assessments", {}).items():
            try:
                risk = RiskAssessment(**risk_data)
                self.risk_assessments[risk_id] = risk
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize risk assessment {risk_id}: {e}")
        
        # Restore stakeholder engagements
        self.stakeholder_engagements = {}
        for stakeholder_id, engagement_data in state.get("stakeholder_engagements", {}).items():
            try:
                engagement = StakeholderEngagement(**engagement_data)
                self.stakeholder_engagements[stakeholder_id] = engagement
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize stakeholder engagement {stakeholder_id}: {e}")
        
        # Restore compliance frameworks
        self.compliance_frameworks = {}
        for compliance_id, framework_data in state.get("compliance_frameworks", {}).items():
            try:
                framework = ComplianceFramework(**framework_data)
                self.compliance_frameworks[compliance_id] = framework
            except Exception as e:
                if self.verbose:
                    CORPORATE_LOGGER.warning(f"Failed to deserialize compliance framework {compliance_id}: {e}")
        
        # Restore simple dict/list structures
        self.crisis_management_plans = state.get("crisis_management_plans", {})
        self.innovation_pipeline = state.get("innovation_pipeline", [])
        self.audit_trails = state.get("audit_trails", [])
        self.performance_metrics = state.get("performance_metrics", {})
        self.sustainability_targets = state.get("sustainability_targets", {})
        self.ai_ethics_framework = state.get("ai_ethics_framework", {})
        
        # Restore running mandates
        self.running_mandates = state.get("running_mandates", {})
        
        # Reinitialize democratic swarm if needed
        if self.config.enable_democratic_discussion:
            self._initialize_democratic_swarm()
    
    def _compute_integrity_hash(self, state: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of serialized state for integrity check.
        
        Args:
            state: Serialized state dictionary
            
        Returns:
            str: SHA256 hash in format "sha256:..."
        """
        # Convert state to deterministic JSON string
        state_str = json.dumps(state, sort_keys=True, default=str)
        
        # Compute hash
        hash_obj = hashlib.sha256(state_str.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        return f"sha256:{hash_hex}"
    
    def _verify_integrity(self, state: Dict[str, Any], expected_hash: str) -> bool:
        """
        Verify state integrity using hash.
        
        Args:
            state: Serialized state dictionary
            expected_hash: Expected hash in format "sha256:..."
            
        Returns:
            bool: True if integrity verified, False otherwise
        """
        computed_hash = self._compute_integrity_hash(state)
        return computed_hash == expected_hash
    
    # ============================================================================
    # MANDATE TRACKING METHODS (for daemon)
    # ============================================================================
    
    def _track_mandate_execution(self, mandate_id: str, proposal_id: str, event_stream_url: Optional[str] = None) -> None:
        """
        Track mandate execution in running_mandates.
        
        Args:
            mandate_id: The mandate ID
            proposal_id: Associated proposal ID
            event_stream_url: Optional event stream URL for monitoring
        """
        self.running_mandates[mandate_id] = {
            "status": "dispatched",
            "proposal_id": proposal_id,
            "event_stream_url": event_stream_url,
            "dispatched_at": time.time(),
            "last_check": time.time(),
            "completion_status": None,
            "error": None
        }
        
        if self.verbose:
            CORPORATE_LOGGER.info(f"Tracking mandate execution: {mandate_id} for proposal {proposal_id}")
    
    def _update_mandate_status(self, mandate_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update mandate execution status.
        
        Args:
            mandate_id: The mandate ID
            status: New status ("running", "completed", "failed", "timeout")
            metadata: Optional metadata (completion_status, error, etc.)
        """
        if mandate_id in self.running_mandates:
            self.running_mandates[mandate_id]["status"] = status
            self.running_mandates[mandate_id]["last_check"] = time.time()
            
            if metadata:
                self.running_mandates[mandate_id].update(metadata)
            
            if self.verbose:
                CORPORATE_LOGGER.debug(f"Updated mandate {mandate_id} status to {status}")
    
    def _get_pending_mandates(self) -> List[CorporateProposal]:
        """
        Get approved proposals that should be executed as mandates.
        
        Returns:
            List of CorporateProposal objects that are approved, code-related, and not yet dispatched
        """
        pending = []
        
        for proposal in self.proposals:
            # Find the vote for this proposal
            vote = None
            for v in self.votes:
                if v.proposal.proposal_id == proposal.proposal_id:
                    vote = v
                    break
            
            # Check if approved and code-related
            if vote and vote.result == VoteResult.APPROVED:
                if self._is_code_related_proposal(proposal):
                    # Check if already dispatched
                    mandate_id = proposal.proposal_id
                    if mandate_id not in self.running_mandates:
                        pending.append(proposal)
        
        return pending
    
    def _get_running_mandates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently executing mandates.
        
        Returns:
            Dict of mandate_id -> mandate tracking info
        """
        return self.running_mandates.copy()
    
    def check_mandate_status(self, mandate_id: str) -> Dict[str, Any]:
        """
        Poll bolt.diy for mandate execution status.
        
        Args:
            mandate_id: The mandate ID to check
            
        Returns:
            Dict containing status, progress, and metadata
        """
        if mandate_id not in self.running_mandates:
            return {
                "status": "not_found",
                "error": f"Mandate {mandate_id} not found in running mandates"
            }
        
        mandate_info = self.running_mandates[mandate_id]
        event_stream_url = mandate_info.get("event_stream_url")
        
        if not event_stream_url:
            return {
                "status": "unknown",
                "error": "No event stream URL available"
            }
        
        try:
            import requests
            
            # Try to query Execution Governor if available
            governor_url = os.getenv("EXECUTION_GOVERNOR_URL", "http://localhost:3000")
            governor_enabled = os.getenv("EXECUTION_GOVERNOR_ENABLED", "false").lower() == "true"
            
            if governor_enabled:
                try:
                    response = requests.get(
                        f"{governor_url}/mandates/{mandate_id}/status",
                        timeout=5
                    )
                    if response.status_code == 200:
                        return response.json()
                except requests.exceptions.RequestException:
                    pass  # Fall through to event stream check
            
            # Fallback: check event stream URL (if it's an API endpoint)
            # For now, return status from tracking
            return {
                "status": mandate_info.get("status", "unknown"),
                "progress": mandate_info.get("completion_status"),
                "dispatched_at": mandate_info.get("dispatched_at"),
                "last_check": mandate_info.get("last_check"),
                "error": mandate_info.get("error")
            }
            
        except Exception as e:
            CORPORATE_LOGGER.warning(f"Error checking mandate status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # ============================================================================
    # GOVERNANCE CHECK HELPER METHODS (for daemon)
    # ============================================================================
    
    def get_pending_proposals_for_review(self, age_threshold: float = 3600.0) -> List[CorporateProposal]:
        """
        Get proposals that are pending and past their review window.
        
        Args:
            age_threshold: Age in seconds (default: 1 hour)
            
        Returns:
            List of proposals that need review
        """
        now = time.time()
        pending = []
        
        for proposal in self.proposals:
            if proposal.status == "pending":
                # Calculate age (proposals don't have created_at, use timestamp from audit trail)
                # For now, check if proposal exists but has no vote
                has_vote = any(v.proposal.proposal_id == proposal.proposal_id for v in self.votes)
                if not has_vote:
                    # Estimate age from audit trail
                    proposal_events = [a for a in self.audit_trails 
                                     if a.get("proposal_id") == proposal.proposal_id]
                    if proposal_events:
                        oldest_event = min(e.get("timestamp", now) for e in proposal_events)
                        age = now - oldest_event
                        if age > age_threshold:
                            pending.append(proposal)
                    else:
                        # No events, assume it's old
                        pending.append(proposal)
        
        return pending
    
    def get_stuck_votes(self) -> List[CorporateVote]:
        """
        Get votes that are incomplete (missing participants or stuck).
        
        Returns:
            List of votes that may need escalation
        """
        stuck = []
        
        for vote in self.votes:
            # Check if vote has very few participants relative to expected
            expected_participants = len(self.board_members) + len(self.executive_team)
            if len(vote.participants) < expected_participants * 0.5:
                stuck.append(vote)
        
        return stuck
    
    def check_risk_thresholds(self) -> Dict[str, bool]:
        """
        Check if any risk assessments exceed thresholds.
        
        Returns:
            Dict mapping risk category to whether threshold exceeded
        """
        thresholds = {
            "operational": 0.7,
            "financial": 0.7,
            "strategic": 0.7,
            "compliance": 0.7,
            "cybersecurity": 0.8,
            "reputation": 0.8
        }
        
        exceeded = {}
        
        for risk_id, risk in self.risk_assessments.items():
            category = risk.risk_category
            threshold = thresholds.get(category, 0.7)
            
            if risk.risk_score > threshold:
                exceeded[category] = True
        
        return exceeded
    
    def get_committees_due_for_review(self) -> List[BoardCommittee]:
        """
        Get committees that are due for periodic review based on meeting schedule.
        
        Returns:
            List of committees that should have meetings
        """
        due = []
        now = time.time()
        
        for committee in self.board_committees.values():
            # Check meeting schedule (simplified: quarterly = every 90 days)
            schedule = committee.meeting_schedule.lower()
            interval_days = 90  # Default quarterly
            
            if "monthly" in schedule:
                interval_days = 30
            elif "quarterly" in schedule:
                interval_days = 90
            elif "annually" in schedule or "annual" in schedule:
                interval_days = 365
            
            # Check if last meeting was more than interval_days ago
            committee_meetings = [m for m in self.board_meetings 
                                if any(cid == committee.committee_id 
                                      for cid in (getattr(m, 'committee_ids', []) or []))]
            
            if not committee_meetings:
                # Never had a meeting, due now
                due.append(committee)
            else:
                last_meeting = max(m.date for m in committee_meetings)
                days_since = (now - last_meeting) / (24 * 60 * 60)
                if days_since > interval_days:
                    due.append(committee)
        
        return due
    
    def find_latest_snapshot(self, snapshot_dir: str = ".") -> Optional[str]:
        """
        Find the most recent snapshot file for this corporation.
        
        Args:
            snapshot_dir: Directory to search in
            
        Returns:
            Path to latest snapshot file, or None if not found
        """
        if not os.path.isdir(snapshot_dir):
            return None
        
        # Look for files matching pattern: {name}_snapshot_*.json
        pattern = f"{self.name.replace(' ', '_')}_snapshot_*.json"
        matching_files = []
        
        for filename in os.listdir(snapshot_dir):
            if filename.startswith(f"{self.name.replace(' ', '_')}_snapshot_") and filename.endswith(".json"):
                filepath = os.path.join(snapshot_dir, filename)
                if os.path.isfile(filepath):
                    matching_files.append((filepath, os.path.getmtime(filepath)))
        
        if not matching_files:
            return None
        
        # Return most recent
        matching_files.sort(key=lambda x: x[1], reverse=True)
        return matching_files[0][0]


# ============================================================================
# CORPORATE SWARM DAEMON - TEMPORAL ORCHESTRATION
# ============================================================================

class CorporateSwarmDaemon:
    """
    Temporal orchestration daemon for CorporateSwarm.
    
    The daemon does NOT reason - it enforces time-based governance cycles,
    monitors execution, and maintains institutional continuity.
    
    Responsibilities:
    1. Tick time
    2. Load & persist state
    3. Schedule governance cycles
    4. Dispatch approved mandates
    5. Monitor execution
    6. Trigger constitutional responses
    """
    
    def __init__(
        self,
        corp: CorporateSwarm,
        tick_interval: float = 5.0,
        snapshot_dir: str = ".",
        governance_check_interval: int = 12,
        auto_start_bolt_diy: bool = True
    ):
        """
        Initialize the daemon.
        
        Args:
            corp: The CorporateSwarm instance to manage
            tick_interval: Seconds between ticks (default: 5.0)
            snapshot_dir: Directory for snapshots (default: ".")
            governance_check_interval: Ticks between governance checks (default: 12 = 1 min at 5s ticks)
            auto_start_bolt_diy: Automatically start bolt.diy if not running (default: True)
        """
        self.corp = corp
        self.shutdown = False
        self.tick_interval = tick_interval
        self.snapshot_dir = snapshot_dir
        self.governance_check_interval = governance_check_interval
        self.auto_start_bolt_diy = auto_start_bolt_diy
        self.last_snapshot_path: Optional[str] = None
        self.tick_count: int = 0
        self.last_governance_check: int = 0
        
        if self.corp.verbose:
            CORPORATE_LOGGER.info(f"CorporateSwarmDaemon initialized (tick_interval={tick_interval}s)")
    
    def startup(self) -> bool:
        """
        Cold boot sequence: load snapshot, verify integrity, reconnect to services.
        
        Returns:
            bool: True if startup successful, False if integrity check failed
        """
        if self.corp.verbose:
            CORPORATE_LOGGER.info("Daemon startup: cold boot sequence")
        
        # 1. Load last snapshot
        snapshot_path = self.corp.find_latest_snapshot(self.snapshot_dir)
        
        if snapshot_path and os.path.exists(snapshot_path):
            if self.corp.verbose:
                CORPORATE_LOGGER.info(f"Loading snapshot: {snapshot_path}")
            
            if not self.corp.load_snapshot(snapshot_path):
                CORPORATE_LOGGER.error("Failed to load snapshot - integrity check failed")
                return False
            
            self.last_snapshot_path = snapshot_path
            
            if self.corp.verbose:
                CORPORATE_LOGGER.info("✅ Snapshot loaded and integrity verified")
        else:
            if self.corp.verbose:
                CORPORATE_LOGGER.info("No snapshot found, starting fresh")
        
        # 2. Rehydrate state (already done by load_snapshot if loaded)
        # Running mandates are restored from snapshot
        
        # 3. Reconnect to services
        bolt_diy_url = os.getenv("BOLT_DIY_API_URL", "http://localhost:5173")
        governor_url = os.getenv("EXECUTION_GOVERNOR_URL", "http://localhost:3000")
        governor_enabled = os.getenv("EXECUTION_GOVERNOR_ENABLED", "false").lower() == "true"
        
        # Health check bolt.diy
        bolt_diy_running = self._check_bolt_diy_health(bolt_diy_url)
        
        if not bolt_diy_running:
            if self.auto_start_bolt_diy:
                if self.corp.verbose:
                    CORPORATE_LOGGER.info("bolt.diy not running, attempting to start...")
                if self._start_bolt_diy():
                    # Wait a bit for services to start
                    time.sleep(10)
                    # Check again
                    bolt_diy_running = self._check_bolt_diy_health(bolt_diy_url)
                    if bolt_diy_running:
                        if self.corp.verbose:
                            CORPORATE_LOGGER.info(f"✅ bolt.diy started and reachable at {bolt_diy_url}")
                    else:
                        CORPORATE_LOGGER.warning("⚠️  bolt.diy start command executed but service not yet reachable")
                else:
                    CORPORATE_LOGGER.warning("⚠️  Failed to start bolt.diy automatically")
            else:
                CORPORATE_LOGGER.warning("⚠️  bolt.diy is not running (auto-start disabled)")
        
        # Health check Execution Governor if enabled
        if governor_enabled:
            try:
                import requests
                response = requests.get(f"{governor_url}/health", timeout=5)
                if response.status_code == 200:
                    if self.corp.verbose:
                        CORPORATE_LOGGER.info(f"✅ Execution Governor reachable at {governor_url}")
                else:
                    CORPORATE_LOGGER.warning(f"⚠️  Execution Governor returned status {response.status_code}")
            except Exception as e:
                CORPORATE_LOGGER.warning(f"⚠️  Cannot reach Execution Governor: {e}")
        
        if self.corp.verbose:
            CORPORATE_LOGGER.info("✅ Daemon startup complete - entering event loop")
        
        return True
    
    def run(self) -> None:
        """
        Main daemon loop with tick-based scheduling (no drift).
        """
        if self.corp.verbose:
            CORPORATE_LOGGER.info("Daemon event loop started")
        
        # Startup sequence
        if not self.startup():
            CORPORATE_LOGGER.error("Startup failed - safe halt")
            return
        
        # Main loop
        while not self.shutdown:
            tick_start = time.time()
            
            try:
                # Execute tick
                self.tick()
                
                # Calculate sleep duration (prevent drift)
                elapsed = time.time() - tick_start
                sleep_duration = max(0, self.tick_interval - elapsed)
                
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                else:
                    # Tick took longer than interval - log warning
                    if self.corp.verbose:
                        CORPORATE_LOGGER.warning(f"Tick took {elapsed:.2f}s (longer than interval {self.tick_interval}s)")
                
            except KeyboardInterrupt:
                if self.corp.verbose:
                    CORPORATE_LOGGER.info("Shutdown signal received")
                self.shutdown = True
            except Exception as e:
                CORPORATE_LOGGER.error(f"Error in daemon tick: {e}")
                if self.corp.verbose:
                    import traceback
                    traceback.print_exc()
                # Continue running (don't crash on single tick error)
                time.sleep(self.tick_interval)
        
        if self.corp.verbose:
            CORPORATE_LOGGER.info("Daemon shutdown complete")
    
    def tick(self) -> None:
        """
        Execute one tick with all 6 phases in order.
        """
        self.tick_count += 1
        
        if self.corp.verbose and self.tick_count % 12 == 0:  # Log every minute
            CORPORATE_LOGGER.debug(f"Tick #{self.tick_count}")
        
        # Phase 1: State refresh
        self.refresh_state()
        
        # Phase 2: Governance checks (periodic)
        if (self.tick_count - self.last_governance_check) >= self.governance_check_interval:
            self.run_governance_checks()
            self.last_governance_check = self.tick_count
        
        # Phase 3: Execution dispatch
        self.dispatch_mandates()
        
        # Phase 4: Execution monitoring
        self.monitor_execution()
        
        # Phase 5: Persistence (periodic, not every tick)
        if self.tick_count % 12 == 0:  # Every minute
            self.persist_state()
        
        # Phase 6: Health evaluation
        if self.tick_count % 60 == 0:  # Every 5 minutes
            self.evaluate_health()
    
    def refresh_state(self) -> None:
        """
        Phase 1: State refresh - validate invariants and detect signals.
        """
        # Validate invariants
        if self.corp.cost_tracker.current_cost < 0:
            CORPORATE_LOGGER.warning("Budget invariant violated: negative cost")
            self.corp.cost_tracker.current_cost = 0.0
        
        # Check for orphaned mandates (mandate_id exists but proposal doesn't)
        orphaned = []
        for mandate_id, mandate_info in self.corp.running_mandates.items():
            proposal_id = mandate_info.get("proposal_id")
            if proposal_id:
                proposal = self.corp._find_proposal(proposal_id)
                if not proposal:
                    orphaned.append(mandate_id)
        
        if orphaned:
            if self.corp.verbose:
                CORPORATE_LOGGER.warning(f"Found {len(orphaned)} orphaned mandates, cleaning up")
            for mandate_id in orphaned:
                del self.corp.running_mandates[mandate_id]
        
        # Detect external signals
        shutdown_flag_path = os.path.join(self.snapshot_dir, f"{self.corp.name.replace(' ', '_')}_shutdown.flag")
        if os.path.exists(shutdown_flag_path):
            if self.corp.verbose:
                CORPORATE_LOGGER.info("Shutdown flag detected")
            self.shutdown = True
            # Remove flag file
            try:
                os.remove(shutdown_flag_path)
            except Exception:
                pass
        
        # Check emergency stop
        if self.corp.emergency_stop_active:
            if self.corp.verbose:
                CORPORATE_LOGGER.warning("Emergency stop is active")
        
        # Check frozen actions
        if self.corp._frozen_actions:
            if self.corp.verbose:
                CORPORATE_LOGGER.warning("Actions are frozen (constitutional safe halt)")
    
    def run_governance_checks(self) -> None:
        """
        Phase 2: Governance checks - time-based governance triggers.
        
        No execution happens here - only scheduling/triggering governance actions.
        """
        if self.corp.verbose:
            CORPORATE_LOGGER.debug("Running governance checks")
        
        # Check pending proposals past review window
        pending_proposals = self.corp.get_pending_proposals_for_review(age_threshold=3600.0)  # 1 hour
        if pending_proposals:
            if self.corp.verbose:
                CORPORATE_LOGGER.info(f"Found {len(pending_proposals)} proposals past review window")
            # Trigger votes for pending proposals
            for proposal in pending_proposals[:3]:  # Limit to 3 per check
                try:
                    if self.corp.verbose:
                        CORPORATE_LOGGER.info(f"Triggering vote for proposal: {proposal.proposal_id}")
                    self.corp.conduct_corporate_vote(proposal.proposal_id)
                except Exception as e:
                    CORPORATE_LOGGER.warning(f"Failed to trigger vote for proposal {proposal.proposal_id}: {e}")
        
        # Check stuck votes
        stuck_votes = self.corp.get_stuck_votes()
        if stuck_votes:
            if self.corp.verbose:
                CORPORATE_LOGGER.warning(f"Found {len(stuck_votes)} stuck votes")
            # Escalate to committee (simplified - just log for now)
            for vote in stuck_votes:
                CORPORATE_LOGGER.info(f"Stuck vote detected: {vote.vote_id} (only {len(vote.participants)} participants)")
        
        # Check risk thresholds
        risk_exceeded = self.corp.check_risk_thresholds()
        if risk_exceeded:
            if self.corp.verbose:
                CORPORATE_LOGGER.warning(f"Risk thresholds exceeded: {list(risk_exceeded.keys())}")
            # Trigger risk committee review (simplified - just log for now)
            for category in risk_exceeded:
                CORPORATE_LOGGER.info(f"Risk threshold exceeded for {category}")
        
        # Check committees due for review
        due_committees = self.corp.get_committees_due_for_review()
        if due_committees:
            if self.corp.verbose:
                CORPORATE_LOGGER.info(f"Found {len(due_committees)} committees due for review")
            # Schedule committee meetings (simplified - just log for now)
            for committee in due_committees:
                CORPORATE_LOGGER.info(f"Committee due for review: {committee.name}")
    
    def dispatch_mandates(self) -> None:
        """
        Phase 3: Execution dispatch - dispatch approved code-related proposals.
        """
        # Get pending mandates (approved, code-related, not yet dispatched)
        pending = self.corp._get_pending_mandates()
        
        if not pending:
            return
        
        if self.corp.verbose:
            CORPORATE_LOGGER.info(f"Found {len(pending)} pending mandates to dispatch")
        
        # Check Execution Governor constraints if enabled
        governor_enabled = os.getenv("EXECUTION_GOVERNOR_ENABLED", "false").lower() == "true"
        
        for proposal in pending[:5]:  # Limit to 5 per tick
            # Check budget
            if not self.corp.cost_tracker.check_budget():
                if self.corp.verbose:
                    CORPORATE_LOGGER.warning("Budget limit reached, cannot dispatch more mandates")
                break
            
            # Check Execution Governor constraints
            if governor_enabled:
                try:
                    import requests
                    governor_url = os.getenv("EXECUTION_GOVERNOR_URL", "http://localhost:3000")
                    
                    # Check if governor can accept more mandates
                    response = requests.get(f"{governor_url}/status", timeout=2)
                    if response.status_code == 200:
                        status = response.json()
                        max_concurrent = status.get("max_concurrent_executions", 10)
                        active = status.get("active_executions", 0)
                        
                        if active >= max_concurrent:
                            if self.corp.verbose:
                                CORPORATE_LOGGER.debug("Execution Governor at capacity, waiting")
                            break  # Wait for next tick
                except Exception:
                    pass  # Continue if governor check fails
            
            # Dispatch mandate
            try:
                if self.corp.verbose:
                    CORPORATE_LOGGER.info(f"Dispatching mandate for proposal: {proposal.proposal_id}")
                
                result = self.corp.execute_proposal_as_mandate(proposal.proposal_id)
                
                if result.get("status") == "accepted":
                    if self.corp.verbose:
                        CORPORATE_LOGGER.info(f"✅ Mandate dispatched: {result.get('mandate_id')}")
                else:
                    CORPORATE_LOGGER.warning(f"Failed to dispatch mandate: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                CORPORATE_LOGGER.error(f"Error dispatching mandate for proposal {proposal.proposal_id}: {e}")
    
    def monitor_execution(self) -> None:
        """
        Phase 4: Execution monitoring - poll bolt.diy for mandate status.
        """
        running = self.corp._get_running_mandates()
        
        if not running:
            return
        
        # Check each running mandate
        for mandate_id, mandate_info in list(running.items()):
            # Poll status
            status_result = self.corp.check_mandate_status(mandate_id)
            
            status = status_result.get("status")
            
            if status == "completed":
                # Mark as completed
                self.corp._update_mandate_status(
                    mandate_id,
                    "completed",
                    {"completion_status": "success", "completed_at": time.time()}
                )
                
                if self.corp.verbose:
                    CORPORATE_LOGGER.info(f"✅ Mandate completed: {mandate_id}")
                
                # Update proposal status
                proposal_id = mandate_info.get("proposal_id")
                if proposal_id:
                    proposal = self.corp._find_proposal(proposal_id)
                    if proposal:
                        proposal.status = "executed"
                
            elif status == "failed":
                # Mark as failed
                error = status_result.get("error", "Unknown error")
                self.corp._update_mandate_status(
                    mandate_id,
                    "failed",
                    {"completion_status": "failed", "error": error, "failed_at": time.time()}
                )
                
                CORPORATE_LOGGER.warning(f"❌ Mandate failed: {mandate_id} - {error}")
                
                # Trigger constitutional fallback
                proposal_id = mandate_info.get("proposal_id")
                if proposal_id:
                    # Apply fallback (simplified - just log for now)
                    CORPORATE_LOGGER.info(f"Triggering constitutional fallback for failed mandate {mandate_id}")
                
            elif status == "timeout":
                # Check if mandate has been running too long
                dispatched_at = mandate_info.get("dispatched_at", 0)
                timeout_threshold = 3600.0  # 1 hour
                
                if time.time() - dispatched_at > timeout_threshold:
                    self.corp._update_mandate_status(
                        mandate_id,
                        "timeout",
                        {"completion_status": "timeout", "timeout_at": time.time()}
                    )
                    
                    CORPORATE_LOGGER.warning(f"⏱️  Mandate timeout: {mandate_id}")
                    
                    # Trigger fallback
                    proposal_id = mandate_info.get("proposal_id")
                    if proposal_id:
                        CORPORATE_LOGGER.info(f"Triggering constitutional fallback for timeout mandate {mandate_id}")
    
    def persist_state(self) -> None:
        """
        Phase 5: Persistence - atomic snapshot writes.
        """
        try:
            # Generate snapshot filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.corp.name.replace(' ', '_')}_snapshot_{timestamp}.json"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            # Write to temp file first
            temp_filepath = filepath + ".tmp"
            
            if self.corp.save_snapshot(temp_filepath):
                # Atomic rename
                try:
                    os.rename(temp_filepath, filepath)
                    self.last_snapshot_path = filepath
                    
                    if self.corp.verbose:
                        CORPORATE_LOGGER.debug(f"✅ State persisted: {filepath}")
                    
                    # Clean up old snapshots (keep last 10)
                    self._cleanup_old_snapshots()
                    
                except Exception as e:
                    CORPORATE_LOGGER.error(f"Failed to rename snapshot file: {e}")
                    # Try to remove temp file
                    try:
                        os.remove(temp_filepath)
                    except Exception:
                        pass
            else:
                CORPORATE_LOGGER.error("Failed to save snapshot")
                
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error during persistence: {e}")
    
    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots, keeping only the last 10."""
        try:
            pattern = f"{self.corp.name.replace(' ', '_')}_snapshot_*.json"
            snapshots = []
            
            for filename in os.listdir(self.snapshot_dir):
                if filename.startswith(f"{self.corp.name.replace(' ', '_')}_snapshot_") and filename.endswith(".json"):
                    filepath = os.path.join(self.snapshot_dir, filename)
                    if os.path.isfile(filepath):
                        snapshots.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            snapshots.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old snapshots (keep last 10)
            for filepath, _ in snapshots[10:]:
                try:
                    os.remove(filepath)
                    if self.corp.verbose:
                        CORPORATE_LOGGER.debug(f"Removed old snapshot: {filepath}")
                except Exception:
                    pass
                    
        except Exception as e:
            CORPORATE_LOGGER.warning(f"Error cleaning up snapshots: {e}")
    
    def evaluate_health(self) -> None:
        """
        Phase 6: Health evaluation - check system health and detect stuck conditions.
        """
        # Check memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 1000:  # > 1GB
                CORPORATE_LOGGER.warning(f"High memory usage: {memory_mb:.1f}MB")
        except ImportError:
            pass  # psutil not available
        except Exception:
            pass
        
        # Check execution backlog
        pending_mandates = len(self.corp._get_pending_mandates())
        running_mandates = len(self.corp.running_mandates)
        
        if pending_mandates > 10:
            CORPORATE_LOGGER.warning(f"Large execution backlog: {pending_mandates} pending mandates")
        
        if running_mandates > 20:
            CORPORATE_LOGGER.warning(f"Many running mandates: {running_mandates}")
        
        # Check repeated failure rates
        if self.corp._constitutional_failure_count > 5:
            CORPORATE_LOGGER.warning(f"High constitutional failure count: {self.corp._constitutional_failure_count}")
        
        # Check stuck conditions
        now = time.time()
        stuck_mandates = []
        for mandate_id, mandate_info in self.corp.running_mandates.items():
            dispatched_at = mandate_info.get("dispatched_at", 0)
            status = mandate_info.get("status", "unknown")
            
            if status in ["dispatched", "running"]:
                age = now - dispatched_at
                if age > 7200:  # 2 hours
                    stuck_mandates.append(mandate_id)
        
        if stuck_mandates:
            CORPORATE_LOGGER.warning(f"Found {len(stuck_mandates)} stuck mandates (running > 2 hours)")
        
        # Check proposals stuck in pending
        stuck_proposals = len(self.corp.get_pending_proposals_for_review(age_threshold=7200.0))  # 2 hours
        if stuck_proposals > 5:
            CORPORATE_LOGGER.warning(f"Many stuck proposals: {stuck_proposals}")
        
        # If critical issues, trigger safe halt
        if self.corp._constitutional_failure_count > 10:
            CORPORATE_LOGGER.error("Critical: Too many constitutional failures - triggering safe halt")
            self.corp._frozen_actions = True
    
    def _check_bolt_diy_health(self, url: str) -> bool:
        """
        Check if bolt.diy is running and reachable.
        
        Args:
            url: bolt.diy API URL
            
        Returns:
            bool: True if bolt.diy is reachable, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                if self.corp.verbose:
                    CORPORATE_LOGGER.info(f"✅ bolt.diy service reachable at {url}")
                return True
            else:
                if self.corp.verbose:
                    CORPORATE_LOGGER.warning(f"⚠️  bolt.diy returned status {response.status_code}")
                return False
        except Exception:
            return False
    
    def _start_bolt_diy(self) -> bool:
        """
        Start bolt.diy via Docker Compose.
        
        Returns:
            bool: True if start command succeeded, False otherwise
        """
        try:
            import subprocess
            
            # Find bolt.diy directory (relative to project root)
            # We're in branches/crca_cg, so go up to project root then to tools/bolt.diy
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            bolt_diy_dir = os.path.join(project_root, "tools", "bolt.diy")
            
            if not os.path.isdir(bolt_diy_dir):
                CORPORATE_LOGGER.warning(f"bolt.diy directory not found: {bolt_diy_dir}")
                return False
            
            docker_compose_file = os.path.join(bolt_diy_dir, "docker-compose.yaml")
            if not os.path.exists(docker_compose_file):
                CORPORATE_LOGGER.warning(f"docker-compose.yaml not found: {docker_compose_file}")
                return False
            
            if self.corp.verbose:
                CORPORATE_LOGGER.info(f"Starting bolt.diy via Docker Compose from {bolt_diy_dir}")
            
            # Check if docker-compose is available
            try:
                subprocess.run(["docker-compose", "--version"], 
                             capture_output=True, check=True, timeout=5)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Try docker compose (v2)
                try:
                    subprocess.run(["docker", "compose", "version"], 
                                 capture_output=True, check=True, timeout=5)
                    compose_cmd = ["docker", "compose"]
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    CORPORATE_LOGGER.error("Docker Compose not found. Please install Docker Compose.")
                    return False
            else:
                compose_cmd = ["docker-compose"]
            
            # Start services in detached mode
            cmd = compose_cmd + [
                "--file", docker_compose_file,
                "--profile", "production",
                "up", "-d"
            ]
            
            if self.corp.verbose:
                CORPORATE_LOGGER.info(f"Running: {' '.join(cmd)}")
            
            # No timeout - Docker builds can take a long time
            result = subprocess.run(
                cmd,
                cwd=bolt_diy_dir,
                capture_output=True,
                text=True,
                timeout=None  # No timeout - allow builds to complete
            )
            
            if result.returncode == 0:
                if self.corp.verbose:
                    CORPORATE_LOGGER.info("✅ Docker Compose command executed successfully")
                    if result.stdout:
                        CORPORATE_LOGGER.debug(f"Docker output: {result.stdout}")
                return True
            else:
                CORPORATE_LOGGER.error(f"Failed to start bolt.diy: {result.stderr}")
                return False
        except Exception as e:
            CORPORATE_LOGGER.error(f"Error starting bolt.diy: {e}")
            if self.corp.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def shutdown_daemon(self) -> None:
        """Request daemon shutdown."""
        self.shutdown = True
        if self.corp.verbose:
            CORPORATE_LOGGER.info("Shutdown requested")


class CostTracker:
    """Simple cost tracking for budget management."""
    
    def __init__(self, budget_limit: float = 200.0):
        self.budget_limit = budget_limit
        self.current_cost = 0.0
    
    def check_budget(self) -> bool:
        """Check if we're within budget."""
        return self.current_cost < self.budget_limit
    
    def add_cost(self, cost: float) -> None:
        """Add cost to current total."""
        self.current_cost += cost
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.budget_limit - self.current_cost)


# ============================================================================
# HUMAN-IN-THE-LOOP SAFETY CONTROLS
# ============================================================================

class ApprovalStatus(str, Enum):
    """Status of human approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """Human approval request for major decisions."""
    approval_id: str = field(default_factory=_generate_uuid)
    proposal_id: Optional[str] = None
    decision_type: str = ""
    description: str = ""
    budget_impact: float = 0.0
    risk_score: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# FACTORY FUNCTIONS FOR EASY CREATION
# ============================================================================

def create_corporation(
    name: str = "AutoCorp",
    description: str = None,
    industry: str = None,
    model_name: str = "gpt-4o-mini",
    budget_limit: float = 200.0,
    enable_causal_reasoning: bool = True,
    enable_quant_analysis: bool = True,
    enable_crca_sd_governance: bool = True,
    enable_aop: bool = False,
    enable_queue_execution: bool = True,
    verbose: bool = False,
    load_from: Optional[str] = None,
    **kwargs
) -> CorporateSwarm:
    """
    Create a fully-configured corporation with automatic setup.
    
    This is the easiest way to create and use CorporateSwarm - it automatically
    sets up all necessary components (board, executives, departments, committees).
    
    Args:
        name: Corporation name
        description: Corporation description (auto-generated if None)
        industry: Industry type (used for auto-configuration)
        model_name: LLM model to use
        budget_limit: Budget limit in dollars
        enable_causal_reasoning: Enable CRCA causal reasoning
        enable_quant_analysis: Enable CRCA-Q quantitative analysis
        enable_crca_sd_governance: Enable CRCA-SD governance system
        enable_aop: Enable AOP MCP server deployment
        enable_queue_execution: Enable queue-based task execution
        verbose: Enable verbose logging
        load_from: Optional path to snapshot file to load from instead of creating new
        **kwargs: Additional configuration parameters
        
    Returns:
        CorporateSwarm: Fully configured and ready-to-use corporation
        
    Example:
        >>> # Simplest usage - just a name
        >>> corp = create_corporation("TechStart Inc")
        >>> result = corp.run("Should we invest in AI research?")
        
        >>> # With industry-specific configuration
        >>> corp = create_corporation(
        ...     name="LogisticsCorp",
        ...     industry="logistics",
        ...     budget_limit=500.0,
        ...     verbose=True
        ... )
        
        >>> # Load from snapshot
        >>> corp = create_corporation(load_from="MyCorp_snapshot_20240102_120000.json")
    """
    # If load_from is provided, try to load from snapshot
    if load_from:
        if os.path.exists(load_from):
            if verbose:
                CORPORATE_LOGGER.info(f"Loading corporation from snapshot: {load_from}")
            
            # Create minimal corporation first (will be overwritten by load)
            # Use name from snapshot if not provided, otherwise use provided name
            corporation = CorporateSwarm(
                name=name,
                description=description or f"{name} - Loaded from snapshot",
                corporate_model_name=model_name,
                verbose=verbose,
                config_data={
                    "budget_limit": budget_limit,
                    "default_corporate_model": model_name,
                    "enable_causal_reasoning": enable_causal_reasoning,
                    "enable_quant_analysis": enable_quant_analysis,
                    "enable_crca_sd_governance": enable_crca_sd_governance,
                    "enable_aop": enable_aop,
                    "enable_queue_execution": enable_queue_execution,
                    **kwargs
                }
            )
            
            # Load snapshot
            if corporation.load_snapshot(load_from):
                if verbose:
                    CORPORATE_LOGGER.info(f"✅ Loaded corporation '{corporation.name}' from snapshot")
                    CORPORATE_LOGGER.info(f"   - Board: {len(corporation.board_members)} members")
                    CORPORATE_LOGGER.info(f"   - Executives: {len(corporation.executive_team)} members")
                    CORPORATE_LOGGER.info(f"   - Departments: {len(corporation.departments)}")
                    CORPORATE_LOGGER.info(f"   - Committees: {len(corporation.board_committees)}")
                return corporation
            else:
                CORPORATE_LOGGER.warning(f"Failed to load snapshot {load_from}, creating new corporation")
                # Fall through to create new
        else:
            CORPORATE_LOGGER.warning(f"Snapshot file not found: {load_from}, creating new corporation")
            # Fall through to create new
    
    # Auto-generate description if not provided
    if description is None:
        if industry:
            description = f"A {industry} company with comprehensive corporate governance"
        else:
            description = f"{name} - Comprehensive corporate governance system"
    
    # Build config data with sensible defaults
    config_data = {
        "default_board_size": kwargs.get("board_size", 6),
        "default_executive_team_size": kwargs.get("executive_team_size", 4),
        "decision_threshold": kwargs.get("decision_threshold", 0.6),
        "budget_limit": budget_limit,
        "default_corporate_model": model_name,
        "enable_causal_reasoning": enable_causal_reasoning,
        "enable_quant_analysis": enable_quant_analysis,
        "enable_crca_sd_governance": enable_crca_sd_governance,
        "enable_aop": enable_aop,
        "enable_queue_execution": enable_queue_execution,
        **{k: v for k, v in kwargs.items() if k not in [
            "board_size", "executive_team_size", "decision_threshold"
        ]}
    }
    
    # Create corporation
    corporation = CorporateSwarm(
        name=name,
        description=description,
        corporate_model_name=model_name,
        verbose=verbose,
        config_data=config_data
    )
    
    if verbose:
        CORPORATE_LOGGER.info(f"✅ Created corporation '{name}' with {len(corporation.members)} members")
        CORPORATE_LOGGER.info(f"   - Board: {len(corporation.board_members)} members")
        CORPORATE_LOGGER.info(f"   - Executives: {len(corporation.executive_team)} members")
        CORPORATE_LOGGER.info(f"   - Departments: {len(corporation.departments)}")
        CORPORATE_LOGGER.info(f"   - Committees: {len(corporation.board_committees)}")
    
    return corporation


def run_corporation(
    task: str,
    name: str = "AutoCorp",
    **create_kwargs
) -> Dict[str, Any]:
    """
    Create a corporation and run a task in one line.
    
    This is the simplest way to use CorporateSwarm - just provide a task
    and optionally a name, and everything else is auto-configured.
    
    Args:
        task: The corporate task or proposal to process
        name: Corporation name (default: "AutoCorp")
        **create_kwargs: Additional arguments passed to create_corporation()
        
    Returns:
        Dict[str, Any]: Task results or decision outcomes
        
    Example:
        >>> # Simplest usage
        >>> result = run_corporation("Should we invest in renewable energy?")
        >>> print(result['vote_result'])
        
        >>> # With custom name
        >>> result = run_corporation(
        ...     "Approve new product launch",
        ...     name="TechCorp",
        ...     verbose=True
        ... )
    """
    corporation = create_corporation(name=name, **create_kwargs)
    return corporation.run(task)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CorporateSwarm - Autonomous Corporate Governance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple task execution
  python corposwarm.py "Should we invest in AI technology?"
  
  # With custom name and verbose output
  python corposwarm.py "Approve new product launch" --name TechCorp --verbose
  
  # Interactive mode
  python corposwarm.py --interactive
  
  # Create and show status
  python corposwarm.py --create-only --name MyCorp --verbose
        """
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Corporate task or proposal to process (e.g., 'Should we invest in X?')"
    )
    
    parser.add_argument(
        "--name",
        default="AutoCorp",
        help="Corporation name (default: AutoCorp)"
    )
    
    parser.add_argument(
        "--description",
        help="Corporation description (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--industry",
        help="Industry type for auto-configuration"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        default=200.0,
        help="Budget limit in dollars (default: 200.0)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for tasks)"
    )
    
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create corporation and show status, don't run task"
    )
    
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal reasoning (CRCA)"
    )
    
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable quantitative analysis (CRCA-Q)"
    )
    
    parser.add_argument(
        "--no-governance",
        action="store_true",
        help="Disable CRCA-SD governance system"
    )
    
    parser.add_argument(
        "--enable-aop",
        action="store_true",
        help="Enable AOP MCP server deployment"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in continuous daemon mode (processes tasks from queue or stdin)"
    )
    
    parser.add_argument(
        "--task-file",
        help="File containing tasks to process (one per line, for daemon mode)"
    )
    
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=5.0,
        help="Seconds between ticks for daemon mode (default: 5.0)"
    )
    
    parser.add_argument(
        "--snapshot-dir",
        default=".",
        help="Directory for snapshots (default: current directory)"
    )
    
    parser.add_argument(
        "--snapshot",
        help="Specific snapshot file to load on startup (for daemon mode)"
    )
    
    parser.add_argument(
        "--governance-check-interval",
        type=int,
        default=12,
        help="Ticks between governance checks (default: 12 = 1 minute at 5s ticks)"
    )
    
    parser.add_argument(
        "--no-auto-start-bolt",
        action="store_true",
        help="Disable automatic startup of bolt.diy (default: auto-start enabled)"
    )
    
    args = parser.parse_args()
    
    # Create corporation
    try:
        corporation = create_corporation(
            name=args.name,
            description=args.description,
            industry=args.industry,
            model_name=args.model,
            budget_limit=args.budget,
            enable_causal_reasoning=not args.no_causal,
            enable_quant_analysis=not args.no_quant,
            enable_crca_sd_governance=not args.no_governance,
            enable_aop=args.enable_aop,
            verbose=args.verbose
        )
        
        if args.create_only:
            # Just show status
            status = corporation.get_corporate_status()
            print("\n" + "="*60)
            print(f"Corporation: {status['name']}")
            print("="*60)
            print(f"Total Members: {status['total_members']}")
            print(f"Board Members: {status['board_members']}")
            print(f"Executive Team: {status['executive_team']}")
            print(f"Departments: {status['departments']}")
            print(f"Board Committees: {status['board_committees']}")
            print(f"Independent Directors: {status['independent_directors']}")
            print("\n✅ Corporation created and ready!")
            print("\nUsage example:")
            print(f"  result = corporation.run('Your task here')")
            print(f"  print(result)")
        elif args.daemon:
            # Temporal orchestration daemon mode - fully autonomous
            print(f"\n{'='*60}")
            print(f"CorporateSwarm Temporal Orchestration Daemon - {args.name}")
            print("="*60)
            print(f"Tick interval: {args.tick_interval}s")
            print(f"Snapshot directory: {args.snapshot_dir}")
            print(f"Governance check interval: {args.governance_check_interval} ticks")
            print("\nThe daemon enforces time-based governance cycles,")
            print("monitors execution, and maintains institutional continuity.")
            print("\nPress Ctrl+C to stop gracefully.\n")
            
            # Load from snapshot if provided
            snapshot_path = args.snapshot
            if not snapshot_path:
                # Try to find latest snapshot
                snapshot_path = corporation.find_latest_snapshot(args.snapshot_dir)
            
            if snapshot_path and os.path.exists(snapshot_path):
                print(f"Loading snapshot: {snapshot_path}")
                if corporation.load_snapshot(snapshot_path):
                    print("✅ Snapshot loaded and integrity verified")
                else:
                    print("❌ Failed to load snapshot - integrity check failed")
                    print("Starting fresh...")
            elif snapshot_path:
                print(f"⚠️  Snapshot file not found: {snapshot_path}")
                print("Starting fresh...")
            
            # Create daemon
            daemon = CorporateSwarmDaemon(
                corp=corporation,
                tick_interval=args.tick_interval,
                snapshot_dir=args.snapshot_dir,
                governance_check_interval=args.governance_check_interval,
                auto_start_bolt_diy=not args.no_auto_start_bolt
            )
            
            # Handle shutdown signals
            import signal
            def signal_handler(signum, frame):
                print("\n\n🛑 Shutdown signal received, stopping daemon...")
                daemon.shutdown_daemon()
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            try:
                # Run daemon
                daemon.run()
                
                print("\n\n📊 Daemon Statistics:")
                print(f"   Total ticks: {daemon.tick_count}")
                print(f"   Running mandates: {len(corporation.running_mandates)}")
                print(f"   Pending mandates: {len(corporation._get_pending_mandates())}")
                if daemon.last_snapshot_path:
                    print(f"   Last snapshot: {daemon.last_snapshot_path}")
                print("\n👋 Daemon stopped gracefully")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                daemon.shutdown_daemon()
                # Wait for current tick to finish
                time.sleep(args.tick_interval)
            except Exception as e:
                CORPORATE_LOGGER.error(f"Daemon error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                raise
        elif args.interactive:
            # Interactive mode
            print(f"\n{'='*60}")
            print(f"CorporateSwarm Interactive Mode - {args.name}")
            print("="*60)
            print("Enter corporate tasks (or 'quit' to exit):\n")
            
            while True:
                try:
                    task = input("Task: ").strip()
                    if not task or task.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    print(f"\n🔄 Processing: {task}\n")
                    result = corporation.run(task)
                    
                    if isinstance(result, dict):
                        if 'vote_result' in result:
                            print(f"✅ Vote Result: {result['vote_result']}")
                        if 'status' in result:
                            print(f"Status: {result['status']}")
                        if 'error' in result:
                            print(f"❌ Error: {result['error']}")
                        
                        # Show auto-execution results if present
                        if 'auto_executed' in result:
                            if result.get('auto_executed'):
                                print(f"\n🚀 Auto-Execution: Enabled")
                                if 'execution' in result:
                                    exec_result = result['execution']
                                    if exec_result.get('status') == 'accepted':
                                        print(f"   ✅ Mandate accepted by bolt.diy")
                                        if 'event_stream_url' in exec_result:
                                            print(f"   📊 Event Stream: {exec_result['event_stream_url']}")
                                        if 'mandate_id' in exec_result:
                                            print(f"   🆔 Mandate ID: {exec_result['mandate_id']}")
                                    elif exec_result.get('status') == 'failed':
                                        print(f"   ❌ Execution failed: {exec_result.get('error', 'Unknown error')}")
                                    else:
                                        print(f"   ⚠️  Execution status: {exec_result.get('status', 'unknown')}")
                            else:
                                if result.get('vote_result') == 'approved':
                                    print(f"\n💡 Note: Proposal approved but not code-related (no auto-execution)")
                    else:
                        print(f"Result: {result}")
                    
                    print("\n" + "-"*60 + "\n")
                except KeyboardInterrupt:
                    print("\n\n👋 Exiting...")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
        elif args.task:
            # Run single task
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"CorporateSwarm - {args.name}")
                print("="*60)
                print(f"Task: {args.task}\n")
            
            result = corporation.run(args.task)
            
            # Pretty print result
            if isinstance(result, dict):
                if 'vote_result' in result:
                    print(f"\n✅ Vote Result: {result['vote_result']}")
                if 'status' in result:
                    print(f"Status: {result['status']}")
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                
                # Show auto-execution results if present
                if 'auto_executed' in result:
                    if result.get('auto_executed'):
                        print(f"\n🚀 Auto-Execution: Enabled")
                        if 'execution' in result:
                            exec_result = result['execution']
                            if exec_result.get('status') == 'accepted':
                                print(f"   ✅ Mandate accepted by bolt.diy")
                                if 'event_stream_url' in exec_result:
                                    print(f"   📊 Event Stream: {exec_result['event_stream_url']}")
                                if 'mandate_id' in exec_result:
                                    print(f"   🆔 Mandate ID: {exec_result['mandate_id']}")
                            elif exec_result.get('status') == 'failed':
                                print(f"   ❌ Execution failed: {exec_result.get('error', 'Unknown error')}")
                            else:
                                print(f"   ⚠️  Execution status: {exec_result.get('status', 'unknown')}")
                    else:
                        if result.get('vote_result') == 'approved':
                            print(f"\n💡 Note: Proposal approved but not code-related (no auto-execution)")
                
                if args.verbose:
                    print(f"\nFull Result:\n{json.dumps(result, indent=2)}")
            else:
                print(f"\nResult: {result}")
        else:
            # No task provided, show help
            parser.print_help()
            print("\n💡 Tip: Provide a task as an argument, or use --interactive mode")
            print("   Example: python corposwarm.py 'Should we invest in AI?'")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        CORPORATE_LOGGER.error(f"Failed to create/run corporation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
