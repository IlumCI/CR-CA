"""
CRCA-SD TUI: Comprehensive Interactive Terminal User Interface

Provides a rich, fully interactive terminal interface for monitoring and controlling
CRCA-SD systems in real-time.

Features:
- Live updating dashboard
- Interactive state visualization
- Policy creation and modification
- Vision progress tracking
- Board management
- Alert monitoring
- Execution control
- Full keyboard navigation and interaction
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import time
import sys
import select
import os
import threading
import queue
import termios
import tty
from dataclasses import dataclass, field

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.columns import Columns
    from rich.align import Align
    from rich import box
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Layout = None
    Panel = None
    Table = None
    Text = None
    Live = None
    Progress = None
    Columns = None
    Align = None
    box = None
    Markdown = None
    Prompt = None
    Confirm = None

from loguru import logger

from crca_sd.crca_sd_core import StateVector, ControlVector

# Import formatter from utils
try:
    from utils.formatter import Formatter
    FORMATTER_AVAILABLE = True
except ImportError:
    Formatter = None
    FORMATTER_AVAILABLE = False


class ViewMode(str, Enum):
    """TUI view modes."""
    DASHBOARD = "dashboard"
    STATE = "state"
    POLICY = "policy"
    VISION = "vision"
    BOARDS = "boards"
    EXECUTIONS = "executions"
    ALERTS = "alerts"
    APPROVALS = "approvals"


class InteractionMode(str, Enum):
    """Interaction modes for the TUI."""
    VIEW = "view"
    SELECT = "select"
    CREATE = "create"
    EDIT = "edit"
    COMMAND = "command"


@dataclass
class TUIState:
    """State for TUI display."""
    current_state: Optional[StateVector] = None
    vision_progress: Optional[Dict[str, float]] = None
    vision_target: Optional[StateVector] = None
    system_status: Optional[Dict[str, Any]] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_approvals: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    policy: Optional[ControlVector] = None
    boards: List[Any] = field(default_factory=list)
    governance_system: Optional[Any] = None
    view_mode: ViewMode = ViewMode.DASHBOARD
    interaction_mode: InteractionMode = InteractionMode.VIEW
    selected_index: int = 0
    selected_execution_id: Optional[str] = None
    selected_approval_id: Optional[str] = None
    selected_board_id: Optional[str] = None
    auto_refresh: bool = True
    refresh_rate: float = 1.0
    last_update: float = field(default_factory=time.time)
    current_command: str = ""
    message: Optional[str] = None
    message_style: str = "white"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.execution_history is None:
            self.execution_history = []
        if self.pending_approvals is None:
            self.pending_approvals = []
        if self.alerts is None:
            self.alerts = []
        if self.boards is None:
            self.boards = []


class CRCA_SD_TUI:
    """
    Comprehensive Interactive Terminal User Interface for CRCA-SD systems.
    
    Provides fully interactive interface with:
    - Current state visualization
    - Vision progress tracking
    - Policy creation and modification
    - Board management
    - Policy execution status
    - Alert monitoring
    - System health
    - Full keyboard navigation
    """
    
    def __init__(self, title: str = "CRCA-SD Control System", use_formatter: bool = True):
        """
        Initialize TUI.
        
        Args:
            title: Title for the TUI
            use_formatter: Whether to use Formatter from utils (default: True)
        """
        if not RICH_AVAILABLE:
            raise ImportError("rich is required for TUI. Install with: pip install rich")
        
        self.console = Console()
        self.title = title
        self.state = TUIState()
        self.is_running = False
        self.start_time = time.time()
        self._should_quit = False
        self._key_handlers: Dict[str, Callable] = {}
        self._key_queue = []
        self._keyboard_thread = None
        self._old_terminal_settings = None
        
        # Use Formatter from utils if available
        if use_formatter and FORMATTER_AVAILABLE and Formatter is not None:
            self.formatter = Formatter(md=True)
            logger.debug("Using Formatter from utils for enhanced formatting")
        else:
            self.formatter = None
        
        self._setup_key_handlers()
    
    def _setup_key_handlers(self) -> None:
        """Setup keyboard handlers."""
        self._key_handlers = {
            'q': self._quit,
            'Q': self._quit,
            'r': self._refresh,
            'R': self._refresh,
            '1': lambda: self._set_view(ViewMode.DASHBOARD),
            '2': lambda: self._set_view(ViewMode.STATE),
            '3': lambda: self._set_view(ViewMode.POLICY),
            '4': lambda: self._set_view(ViewMode.VISION),
            '5': lambda: self._set_view(ViewMode.BOARDS),
            '6': lambda: self._set_view(ViewMode.EXECUTIONS),
            '7': lambda: self._set_view(ViewMode.ALERTS),
            '8': lambda: self._set_view(ViewMode.APPROVALS),
            'a': self._toggle_auto_refresh,
            'A': self._toggle_auto_refresh,
            'c': self._enter_command_mode,
            'C': self._enter_command_mode,
            'n': self._create_new_item,
            'N': self._create_new_item,
            's': self._select_item,
            'S': self._select_item,
            'enter': self._activate_selected,
            '\r': self._activate_selected,
            '\n': self._activate_selected,
            'up': self._navigate_up,
            'down': self._navigate_down,
            'k': self._navigate_up,
            'j': self._navigate_down,
            'h': self._navigate_left,
            'l': self._navigate_right,
            'escape': self._exit_interaction,
            '\x1b': self._exit_interaction,
            '?': self._show_help,
        }
    
    def _quit(self) -> None:
        """Quit the TUI."""
        self._should_quit = True
    
    def _refresh(self) -> None:
        """Manually refresh the display."""
        self.state.message = "Status refreshed"
        self.state.message_style = "green"
        self.state.last_update = time.time()
    
    def _set_view(self, mode: ViewMode) -> None:
        """Set the current view mode."""
        self.state.view_mode = mode
        self.state.selected_index = 0
        self.state.interaction_mode = InteractionMode.VIEW
        self.state.message = None
    
    def _toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh mode."""
        self.state.auto_refresh = not self.state.auto_refresh
        status = "enabled" if self.state.auto_refresh else "disabled"
        self.state.message = f"Auto-refresh {status}"
        self.state.message_style = "green"
    
    def _enter_command_mode(self) -> None:
        """Enter command input mode."""
        self.state.interaction_mode = InteractionMode.COMMAND
        self.state.current_command = ""
    
    def _create_new_item(self) -> None:
        """Create a new item based on current view."""
        if self.state.view_mode == ViewMode.POLICY:
            self._create_policy_interactive()
        else:
            self.state.message = "Cannot create items in this view"
            self.state.message_style = "yellow"
    
    def _select_item(self) -> None:
        """Enter selection mode."""
        if self._get_list_items():
            self.state.interaction_mode = InteractionMode.SELECT
            self.state.message = "Use arrow keys to navigate, Enter to select"
            self.state.message_style = "cyan"
        else:
            self.state.message = "No items to select"
            self.state.message_style = "yellow"
    
    def _activate_selected(self) -> None:
        """Activate the selected item."""
        if self.state.interaction_mode == InteractionMode.SELECT:
            self._show_item_details()
        elif self.state.interaction_mode == InteractionMode.COMMAND:
            self._execute_command()
    
    def _navigate_up(self) -> None:
        """Navigate up in list."""
        items = self._get_list_items()
        if items and self.state.selected_index > 0:
            self.state.selected_index -= 1
    
    def _navigate_down(self) -> None:
        """Navigate down in list."""
        items = self._get_list_items()
        if items and self.state.selected_index < len(items) - 1:
            self.state.selected_index += 1
    
    def _navigate_left(self) -> None:
        """Navigate left (back)."""
        self.state.interaction_mode = InteractionMode.VIEW
        self.state.message = None
    
    def _navigate_right(self) -> None:
        """Navigate right (forward)."""
        if self._get_list_items():
            self._select_item()
    
    def _exit_interaction(self) -> None:
        """Exit current interaction mode."""
        self.state.interaction_mode = InteractionMode.VIEW
        self.state.current_command = ""
        self.state.message = None
    
    def _show_help(self) -> None:
        """Show help information."""
        self.state.message = (
            "Help: [1-8] Views | [N]ew Policy | [S]elect | [C]ommand | "
            "[R]efresh | [A]uto-refresh | [Q]uit | [?] Help"
        )
        self.state.message_style = "cyan"
    
    def _get_list_items(self) -> List[Any]:
        """Get list items for current view."""
        if self.state.view_mode == ViewMode.EXECUTIONS:
            return self.state.execution_history
        elif self.state.view_mode == ViewMode.APPROVALS:
            return self.state.pending_approvals
        elif self.state.view_mode == ViewMode.BOARDS:
            return self.state.boards
        elif self.state.view_mode == ViewMode.ALERTS:
            return self.state.alerts
        return []
    
    def _show_item_details(self) -> None:
        """Show details for selected item."""
        items = self._get_list_items()
        if not items or self.state.selected_index >= len(items):
            return
        
        selected = items[self.state.selected_index]
        
        if self.state.view_mode == ViewMode.EXECUTIONS:
            self.state.selected_execution_id = selected.get("execution_id")
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.APPROVALS:
            self.state.selected_approval_id = selected.get("approval_id")
            self.state.interaction_mode = InteractionMode.VIEW
        elif self.state.view_mode == ViewMode.BOARDS:
            if hasattr(selected, 'board_id'):
                self.state.selected_board_id = selected.board_id
            self.state.interaction_mode = InteractionMode.VIEW
    
    def _execute_command(self) -> None:
        """Execute a command."""
        if not self.state.current_command:
            self.state.interaction_mode = InteractionMode.VIEW
            return
        
        cmd = self.state.current_command.strip().lower()
        parts = cmd.split()
        
        if not parts:
            self.state.interaction_mode = InteractionMode.VIEW
            return
        
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if command == "policy" or command == "create":
                self._create_policy_interactive()
            elif command == "approve" and args:
                self._approve_interactive(args[0])
            elif command == "reject" and args:
                self._reject_interactive(args[0])
            elif command == "status":
                self._show_full_status()
            elif command == "help":
                self._show_help()
            else:
                self.state.message = f"Unknown command: {command}. Type 'help' for commands"
                self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Command error: {str(e)}"
            self.state.message_style = "red"
        finally:
            self.state.current_command = ""
            self.state.interaction_mode = InteractionMode.VIEW
    
    def _create_policy_interactive(self) -> None:
        """Interactively create a policy."""
        try:
            self.console.clear()
            self.console.print(Panel("Create New Policy", border_style="cyan", box=box.ROUNDED))
            
            # Default budget categories
            categories = [
                "education", "infrastructure", "healthcare", 
                "social_welfare", "defense", "other"
            ]
            
            budget_shares = {}
            total = 0.0
            
            self.console.print("\nEnter budget shares (must sum to 1.0):")
            for category in categories:
                share_str = Prompt.ask(f"{category.replace('_', ' ').title()} share", default="0.0")
                try:
                    share = float(share_str)
                    budget_shares[category] = share
                    total += share
                except ValueError:
                    budget_shares[category] = 0.0
            
            # Normalize if needed
            if total > 0 and abs(total - 1.0) > 0.01:
                if Confirm.ask(f"Total is {total:.2%}, normalize to 1.0?"):
                    for category in budget_shares:
                        budget_shares[category] /= total
            
            # Create ControlVector
            policy = ControlVector(budget_shares=budget_shares)
            self.state.policy = policy
            
            self.state.message = "Policy created successfully"
            self.state.message_style = "green"
            
        except KeyboardInterrupt:
            self.state.message = "Policy creation cancelled"
            self.state.message_style = "yellow"
        except Exception as e:
            self.state.message = f"Error creating policy: {str(e)}"
            self.state.message_style = "red"
    
    def _approve_interactive(self, approval_id: str) -> None:
        """Interactively approve an item."""
        self.state.message = f"Approval {approval_id} approved (simulated)"
        self.state.message_style = "green"
    
    def _reject_interactive(self, approval_id: str) -> None:
        """Interactively reject an item."""
        self.state.message = f"Approval {approval_id} rejected (simulated)"
        self.state.message_style = "red"
    
    def _show_full_status(self) -> None:
        """Show full system status."""
        self.state.view_mode = ViewMode.DASHBOARD
        self.state.message = "Full status displayed"
        self.state.message_style = "green"
    
    def create_layout(self) -> Layout:
        """Create the main layout structure."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create header panel."""
        uptime = time.time() - self.start_time
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
        
        last_update_str = datetime.fromtimestamp(self.state.last_update).strftime("%H:%M:%S") if self.state.last_update else "N/A"
        
        header_text = Text()
        header_text.append(self.title, style="bold bright_white")
        header_text.append(" | ", style="dim")
        header_text.append(f"Uptime: {uptime_str}", style="cyan")
        header_text.append(" | ", style="dim")
        header_text.append(f"Last Update: {last_update_str}", style="green")
        header_text.append(" | ", style="dim")
        
        if self.state.auto_refresh:
            header_text.append("Auto-Refresh ON", style="green")
        else:
            header_text.append("Auto-Refresh OFF", style="yellow")
        
        header_text.append(" | ", style="dim")
        header_text.append(f"View: {self.state.view_mode.value.upper()}", style="magenta")
        
        if self.state.interaction_mode != InteractionMode.VIEW:
            header_text.append(" | ", style="dim")
            header_text.append(f"Mode: {self.state.interaction_mode.value.upper()}", style="cyan")
        
        return Panel(Align.center(header_text), style="bold blue", box=box.ROUNDED)
    
    def create_footer(self) -> Layout:
        """Create footer with controls and messages."""
        footer_layout = Layout()
        footer_layout.split_column(
            Layout(name="controls", size=2),
            Layout(name="message", size=2)
        )
        
        # Controls
        controls = Text()
        controls.append("Controls: ", style="bold")
        controls.append("[Q]uit ", style="red")
        controls.append("[R]efresh ", style="yellow")
        controls.append("[1-8] Views ", style="cyan")
        controls.append("[N]ew Policy ", style="green")
        controls.append("[S]elect ", style="blue")
        controls.append("[C]ommand ", style="magenta")
        controls.append("[A]uto-refresh ", style="green")
        controls.append("[?] Help", style="dim")
        
        footer_layout["controls"].update(
            Panel(Align.center(controls), border_style="blue", box=box.ROUNDED)
        )
        
        # Message/Command line
        if self.state.interaction_mode == InteractionMode.COMMAND:
            cmd_text = Text()
            cmd_text.append("Command: ", style="bold cyan")
            cmd_text.append(self.state.current_command, style="white")
            cmd_text.append("_", style="dim")  # Cursor
            footer_layout["message"].update(
                Panel(cmd_text, border_style="cyan", box=box.ROUNDED)
            )
        elif self.state.message:
            msg_text = Text(self.state.message, style=self.state.message_style)
            footer_layout["message"].update(
                Panel(Align.center(msg_text), border_style=self.state.message_style, box=box.ROUNDED)
            )
        else:
            footer_layout["message"].update(
                Panel("", border_style="dim", box=box.ROUNDED)
            )
        
        return footer_layout
    
    def create_state_panel(self) -> Panel:
        """Create state visualization panel."""
        if self.state.current_state is None:
            return Panel("No state data available", title="Current State", border_style="yellow", box=box.ROUNDED)
        
        x = self.state.current_state
        
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        
        # Key metrics
        table.add_row("Population", f"{x.P:,.0f}")
        table.add_row("GDP", f"${x.Y/1e9:.2f}B")
        table.add_row("Unemployment", f"{x.U:.1%}", style="red" if x.U > 0.10 else "green")
        table.add_row("Stability", f"{x.S:.1%}", style="green" if x.S > 0.70 else "yellow" if x.S > 0.50 else "red")
        table.add_row("Literacy", f"{x.literacy:.1%}")
        table.add_row("Infrastructure", f"{x.I:.1%}")
        table.add_row("Capital", f"${x.K/1e9:.2f}B")
        table.add_row("Wage", f"${x.W:.2f}")
        
        return Panel(table, title="Current State", border_style="cyan", box=box.ROUNDED)
    
    def create_vision_panel(self) -> Panel:
        """Create vision progress panel."""
        if self.state.vision_progress is None:
            return Panel("No vision data available", title="Vision Progress", border_style="yellow", box=box.ROUNDED)
        
        progress = self.state.vision_progress
        
        # Overall progress bar
        overall = progress.get("overall", 0.0)
        progress_text = Text()
        progress_text.append("Overall: ", style="dim")
        progress_text.append(f"{overall:.1%}", style="bold bright_green" if overall > 0.7 else "bold yellow" if overall > 0.4 else "bold red")
        
        # Individual metrics
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        
        metrics = [
            ("Unemployment", progress.get("unemployment", 0.0), 0.8),
            ("GDP", progress.get("gdp", 0.0), 0.7),
            ("Stability", progress.get("stability", 0.0), 0.76),
            ("Infrastructure", progress.get("infrastructure", 0.0), 0.76),
            ("Literacy", progress.get("literacy", 0.0), 0.99),
        ]
        
        for name, value, threshold in metrics:
            style = "green" if value >= threshold else "yellow" if value >= threshold * 0.7 else "red"
            table.add_row(name, f"{value:.1%}", style=style)
        
        content = Columns([progress_text, table], equal=True)
        
        return Panel(content, title="Vision Progress", border_style="bright_green", box=box.ROUNDED)
    
    def create_policy_panel(self) -> Panel:
        """Create current policy panel."""
        if self.state.policy is None:
            return Panel("No active policy. Use [N] or command 'policy' to create one.", title="Current Policy", border_style="yellow", box=box.ROUNDED)
        
        policy = self.state.policy
        
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        
        # Sort by budget share
        sorted_shares = sorted(policy.budget_shares.items(), key=lambda x: x[1], reverse=True)
        
        for category, share in sorted_shares[:7]:  # Top 7
            bar_length = int(share * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            style = "green" if share > 0.15 else "yellow" if share > 0.10 else "dim"
            table.add_row(category.capitalize(), f"{bar} {share:.1%}", style=style)
        
        return Panel(table, title="Current Policy", border_style="magenta", box=box.ROUNDED)
    
    def create_status_panel(self) -> Panel:
        """Create system status panel."""
        if self.state.system_status is None:
            return Panel("No status data available", title="System Status", border_style="yellow", box=box.ROUNDED)
        
        status = self.state.system_status
        
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        
        is_running = status.get("is_running", False)
        status_text = "Running" if is_running else "Stopped"
        table.add_row("Status", status_text, style="green" if is_running else "red")
        
        table.add_row("Executions", str(status.get("n_executions", 0)))
        table.add_row("Pending Approvals", str(status.get("n_pending_approvals", 0)),
                     style="yellow" if status.get("n_pending_approvals", 0) > 0 else "dim")
        
        confidence = status.get("state_confidence", 0.0)
        conf_style = "green" if confidence > 0.9 else "yellow" if confidence > 0.7 else "red"
        table.add_row("State Confidence", f"{confidence:.1%}", style=conf_style)
        
        health = status.get("system_health", {})
        health_status = health.get("status", "unknown")
        health_style = "green" if health_status == "healthy" else "yellow" if health_status == "degraded" else "red"
        table.add_row("System Health", health_status.capitalize(), style=health_style)
        
        return Panel(table, title="System Status", border_style="blue", box=box.ROUNDED)
    
    def create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        if not self.state.alerts:
            return Panel("No active alerts", title="Alerts", border_style="green", box=box.ROUNDED)
        
        # Show last 5 alerts
        recent_alerts = self.state.alerts[-5:]
        
        table = Table.grid(padding=(0, 1))
        table.add_column(style="white", width=12)
        table.add_column(style="white")
        
        for idx, alert in enumerate(reversed(recent_alerts)):
            level = alert.get("level", "info").upper()
            message = alert.get("message", "No message")[:50]
            
            if level == "CRITICAL":
                style = "bold red"
                level_text = f"CRITICAL"
            elif level == "WARNING":
                style = "bold yellow"
                level_text = f"WARNING"
            else:
                style = "dim"
                level_text = f"INFO"
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                level_text = f"> {level_text}"
                style = style + " bold"
            
            table.add_row(level_text, message, style=style)
        
        return Panel(table, title="Alerts", border_style="red", box=box.ROUNDED)
    
    def create_history_panel(self) -> Panel:
        """Create execution history panel."""
        if not self.state.execution_history:
            return Panel("No execution history", title="Recent Executions", border_style="dim", box=box.ROUNDED)
        
        # Show last 5 executions
        recent = self.state.execution_history[-5:]
        
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=15)
        table.add_column(style="white")
        
        for idx, exec_item in enumerate(reversed(recent)):
            exec_id = exec_item.get("execution_id", "unknown")[:8]
            status = exec_item.get("status", "unknown")
            automated = exec_item.get("automated", False)
            
            status_style = "green" if status == "executed" else "yellow"
            auto_text = "AUTO" if automated else "MANUAL"
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                exec_id = f"> {exec_id}"
                status_style = status_style + " bold"
            
            table.add_row(f"{auto_text} {exec_id}", status, style=status_style)
        
        return Panel(table, title="Recent Executions", border_style="cyan", box=box.ROUNDED)
    
    def render_executions_view(self) -> Panel:
        """Render detailed executions view."""
        if not self.state.execution_history:
            return Panel("No execution history available", border_style="yellow", box=box.ROUNDED)
        
        executions_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        executions_table.add_column("", width=2)
        executions_table.add_column("Execution ID", style="cyan", width=12)
        executions_table.add_column("Status", style="white", width=12)
        executions_table.add_column("Type", style="yellow", width=10)
        executions_table.add_column("Timestamp", style="dim", width=15)
        
        for idx, exec_item in enumerate(self.state.execution_history[-20:]):
            exec_id = exec_item.get("execution_id", "unknown")[:10]
            status = exec_item.get("status", "unknown")
            automated = exec_item.get("automated", False)
            timestamp = exec_item.get("timestamp", 0)
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            status_style = "green" if status == "executed" else "yellow" if status == "pending" else "red"
            auto_text = "AUTO" if automated else "MANUAL"
            
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else "N/A"
            
            executions_table.add_row(
                selector,
                Text(exec_id, style=row_style) if row_style else exec_id,
                f"[{status_style}]{status.upper()}[/{status_style}]",
                auto_text,
                time_str
            )
        
        return Panel(
            executions_table,
            title="Execution History",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def render_approvals_view(self) -> Panel:
        """Render approvals view."""
        if not self.state.pending_approvals:
            return Panel("No pending approvals", border_style="green", box=box.ROUNDED)
        
        approvals_table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        approvals_table.add_column("", width=2)
        approvals_table.add_column("Approval ID", style="cyan", width=12)
        approvals_table.add_column("Type", style="yellow", width=15)
        approvals_table.add_column("Status", style="white", width=12)
        approvals_table.add_column("Timestamp", style="dim", width=15)
        
        for idx, approval in enumerate(self.state.pending_approvals):
            approval_id = approval.get("approval_id", "unknown")[:10]
            approval_type = approval.get("type", "unknown")
            status = approval.get("status", "pending")
            timestamp = approval.get("timestamp", 0)
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            status_style = "yellow" if status == "pending" else "green" if status == "approved" else "red"
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp else "N/A"
            
            approvals_table.add_row(
                selector,
                Text(approval_id, style=row_style) if row_style else approval_id,
                approval_type,
                f"[{status_style}]{status.upper()}[/{status_style}]",
                time_str
            )
        
        return Panel(
            approvals_table,
            title="Pending Approvals",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def render_boards_view(self) -> Panel:
        """Render boards view."""
        if not self.state.boards:
            return Panel("No boards available", border_style="yellow", box=box.ROUNDED)
        
        boards_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        boards_table.add_column("", width=2)
        boards_table.add_column("Board ID", style="cyan", width=15)
        boards_table.add_column("Type", style="yellow", width=15)
        boards_table.add_column("Members", style="white", justify="right", width=10)
        
        for idx, board in enumerate(self.state.boards):
            board_id = getattr(board, 'board_id', 'unknown')[:12]
            board_type = getattr(board, 'board_type', None)
            members = getattr(board, 'members', [])
            
            if self.state.interaction_mode == InteractionMode.SELECT and idx == self.state.selected_index:
                selector = ">"
                row_style = "bold"
            else:
                selector = " "
                row_style = None
            
            type_str = str(board_type).split('.')[-1] if board_type else "Unknown"
            
            boards_table.add_row(
                selector,
                Text(board_id, style=row_style) if row_style else board_id,
                type_str,
                str(len(members))
            )
        
        return Panel(
            boards_table,
            title="Boards",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def render_dashboard(self) -> Layout:
        """Render the main dashboard view."""
        layout = Layout()
        layout.split_column(
            Layout(name="state", ratio=2),
            Layout(name="vision", ratio=1),
            Layout(name="policy", ratio=1)
        )
        
        layout["state"].update(self.create_state_panel())
        layout["vision"].update(self.create_vision_panel())
        layout["policy"].update(self.create_policy_panel())
        
        return layout
    
    def render(self) -> Layout:
        """Render the complete TUI layout."""
        layout = self.create_layout()
        
        layout["header"].update(self.create_header())
        layout["footer"].update(self.create_footer())
        
        # Render main content based on view mode
        if self.state.view_mode == ViewMode.DASHBOARD:
            main_content = self.render_dashboard()
            layout["left"].update(main_content["state"])
            layout["right"].split_column(
                Layout(main_content["vision"]),
                Layout(main_content["policy"]),
                Layout(self.create_status_panel()),
                Layout(self.create_alerts_panel()),
                Layout(self.create_history_panel())
            )
        elif self.state.view_mode == ViewMode.STATE:
            layout["left"].update(self.create_state_panel())
            layout["right"].update(
                Panel(
                    "State Details\n\nView current economic state metrics",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.POLICY:
            layout["left"].update(self.create_policy_panel())
            layout["right"].update(
                Panel(
                    "Policy Management\n\nPress [N] to create new policy\nPress [C] for command mode",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.VISION:
            layout["left"].update(self.create_vision_panel())
            layout["right"].update(
                Panel(
                    "Vision Progress\n\nTrack progress towards vision goals",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.BOARDS:
            layout["left"].update(self.render_boards_view())
            layout["right"].update(
                Panel(
                    "Board Details\n\nPress [S] to select a board\nPress [Enter] to view details",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.EXECUTIONS:
            layout["left"].update(self.render_executions_view())
            layout["right"].update(
                Panel(
                    "Execution Details\n\nPress [S] to select an execution\nPress [Enter] to view details",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.ALERTS:
            layout["left"].update(self.create_alerts_panel())
            layout["right"].update(
                Panel(
                    "Alert Details\n\nView and manage system alerts",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        elif self.state.view_mode == ViewMode.APPROVALS:
            layout["left"].update(self.render_approvals_view())
            layout["right"].update(
                Panel(
                    "Approval Management\n\nPress [S] to select\nUse commands: approve <id> or reject <id>",
                    border_style="dim",
                    box=box.ROUNDED
                )
            )
        
        return layout
    
    def update_state(
        self,
        current_state: Optional[StateVector] = None,
        vision_progress: Optional[Dict[str, float]] = None,
        vision_target: Optional[StateVector] = None,
        system_status: Optional[Dict[str, Any]] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None,
        pending_approvals: Optional[List[Dict[str, Any]]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
        policy: Optional[ControlVector] = None,
        boards: Optional[List[Any]] = None,
        governance_system: Optional[Any] = None
    ) -> None:
        """
        Update TUI state.
        
        Args:
            current_state: Current state vector
            vision_progress: Progress towards vision
            vision_target: Target vision state
            system_status: System status dict
            execution_history: Execution history
            pending_approvals: Pending approvals
            alerts: Active alerts
            policy: Current policy
            boards: List of boards
            governance_system: Governance system instance
        """
        if current_state is not None:
            self.state.current_state = current_state
        if vision_progress is not None:
            self.state.vision_progress = vision_progress
        if vision_target is not None:
            self.state.vision_target = vision_target
        if system_status is not None:
            self.state.system_status = system_status
        if execution_history is not None:
            self.state.execution_history = execution_history
        if pending_approvals is not None:
            self.state.pending_approvals = pending_approvals
        if alerts is not None:
            self.state.alerts = alerts
        if policy is not None:
            self.state.policy = policy
        if boards is not None:
            self.state.boards = boards
        if governance_system is not None:
            self.state.governance_system = governance_system
        
        self.state.last_update = time.time()
    
    def register_key_handler(self, key: str, handler: Callable) -> None:
        """
        Register a key handler.
        
        Args:
            key: Key to handle (e.g., 'q', 'Q')
            handler: Callback function to call when key is pressed
        """
        self._key_handlers[key.lower()] = handler
    
    def _setup_keyboard_listener(self) -> None:
        """Setup keyboard listener thread."""
        if sys.platform != "win32":
            try:
                # Save terminal settings
                self._old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self._old_terminal_settings = None
        
        self._key_queue = []
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()
    
    def _keyboard_listener(self) -> None:
        """Keyboard listener thread."""
        try:
            while self.is_running and not self._should_quit:
                key = self._get_key()
                if key:
                    # Handle escape sequences for arrow keys
                    if key == '\x1b':
                        try:
                            if sys.platform != "win32":
                                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                                    seq = sys.stdin.read(2)
                                    if seq == '[A':
                                        key = 'up'
                                    elif seq == '[B':
                                        key = 'down'
                                    elif seq == '[C':
                                        key = 'right'
                                    elif seq == '[D':
                                        key = 'left'
                                    else:
                                        key = 'escape'
                                else:
                                    key = 'escape'
                            else:
                                key = 'escape'
                        except Exception:
                            key = 'escape'
                    
                    self._key_queue.append(key)
                time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Keyboard listener error: {e}")
    
    def _get_key(self) -> Optional[str]:
        """Get a key press from stdin (non-blocking)."""
        if sys.platform == "win32":
            try:
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore')
                    return key
            except Exception:
                pass
        else:
            try:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    return key
            except Exception:
                pass
        return None
    
    def _process_keys(self) -> None:
        """Process queued key presses."""
        if not self._key_queue:
            return
        
        while self._key_queue:
            key = self._key_queue.pop(0)
            
            # Handle command mode
            if self.state.interaction_mode == InteractionMode.COMMAND:
                if key == '\r' or key == '\n':
                    self._activate_selected()
                elif key == '\x1b' or key == 'escape':
                    self._exit_interaction()
                elif key == '\x7f' or key == '\b':  # Backspace
                    if self.state.current_command:
                        self.state.current_command = self.state.current_command[:-1]
                elif len(key) == 1 and key.isprintable() and ord(key) >= 32:
                    self.state.current_command += key
                continue
            
            # Handle normal key presses
            handler = self._key_handlers.get(key)
            if handler:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Error handling key {key}: {e}")
    
    def run_live(self, update_callback: Optional[Callable] = None, refresh_rate: float = 1.0) -> None:
        """
        Run TUI with live updates and keyboard input handling.
        
        Args:
            update_callback: Callback function that updates state (called each refresh)
            refresh_rate: Refresh rate in seconds
        """
        self.is_running = True
        self._should_quit = False
        self.state.refresh_rate = refresh_rate
        
        # Setup keyboard listener
        self._setup_keyboard_listener()
        
        try:
            with Live(
                self.render(),
                refresh_per_second=1.0 / refresh_rate,
                screen=True
            ) as live:
                while not self._should_quit:
                    # Process keyboard input
                    self._process_keys()
                    
                    # Update state via callback
                    if update_callback:
                        try:
                            update_callback(self)
                        except Exception as e:
                            logger.debug(f"Error in update callback: {e}")
                    
                    # Clear message after a delay
                    if self.state.message and time.time() - self.state.last_update > 3:
                        self.state.message = None
                    
                    # Render updated layout
                    live.update(self.render())
                    
                    # Sleep briefly to avoid CPU spinning
                    time.sleep(min(refresh_rate, 0.1))
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
            self._should_quit = True
        finally:
            self.is_running = False
            if sys.platform != "win32" and self._old_terminal_settings:
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_terminal_settings)
                except:
                    pass
            self.console.clear()
