"""Rollback system for intervention recovery.

Provides checkpoint creation, intervention recording, and rollback
execution for recovering from failed interventions.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import sqlite3
import json
import uuid
from loguru import logger

from schemas.policy import InterventionSpec


class RollbackManager:
    """Rollback manager for intervention recovery.
    
    Features:
    - State checkpoint creation
    - Intervention recording with results
    - Rollback execution (reverse interventions)
    - Checkpoint restoration
    - Time-based retention
    """
    
    def __init__(self, db_path: str = ":memory:", retention_days: int = 7):
        """
        Initialize rollback manager.
        
        Args:
            db_path: Path to SQLite database (or ":memory:" for in-memory)
            retention_days: Number of days to retain checkpoints
        """
        self.db_path = db_path
        self.retention_days = retention_days
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                epoch INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        
        # Interventions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id TEXT PRIMARY KEY,
                checkpoint_id TEXT,
                epoch INTEGER NOT NULL,
                lever_id TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                result_json TEXT,
                rollback_descriptor_json TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
            )
        """)
        
        # Rollback history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                rollback_id TEXT PRIMARY KEY,
                epoch INTEGER NOT NULL,
                n_steps INTEGER NOT NULL,
                intervention_ids TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def create_checkpoint(self, epoch: int, state: Dict[str, Any]) -> str:
        """
        Create a state checkpoint.
        
        Args:
            epoch: Epoch number
            state: State dictionary to checkpoint
            
        Returns:
            str: Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())
        state_json = json.dumps(state)
        created_at = datetime.now(timezone.utc).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO checkpoints (checkpoint_id, epoch, state_json, created_at)
            VALUES (?, ?, ?, ?)
        """, (checkpoint_id, epoch, state_json, created_at))
        self.conn.commit()
        
        logger.info(f"Created checkpoint {checkpoint_id} for epoch {epoch}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_id
    
    def record_intervention(
        self,
        epoch: int,
        intervention: InterventionSpec,
        result: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None
    ) -> str:
        """
        Record an intervention with its result.
        
        Args:
            epoch: Epoch number
            intervention: Intervention specification
            result: Optional execution result
            checkpoint_id: Optional checkpoint ID associated with this intervention
            
        Returns:
            str: Intervention ID
        """
        intervention_id = str(uuid.uuid4())
        parameters_json = json.dumps(intervention.parameters)
        result_json = json.dumps(result) if result else None
        rollback_descriptor_json = json.dumps(intervention.rollback_descriptor) if intervention.rollback_descriptor else None
        created_at = datetime.now(timezone.utc).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO interventions (
                intervention_id, checkpoint_id, epoch, lever_id,
                parameters_json, result_json, rollback_descriptor_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            intervention_id, checkpoint_id, epoch, intervention.lever_id,
            parameters_json, result_json, rollback_descriptor_json, created_at
        ))
        self.conn.commit()
        
        logger.debug(f"Recorded intervention {intervention_id} for epoch {epoch}")
        
        return intervention_id
    
    def rollback(
        self,
        epoch: int,
        n_steps: int,
        actuator_registry: Optional[Any] = None
    ) -> List[str]:
        """
        Rollback last N interventions.
        
        Args:
            epoch: Current epoch
            n_steps: Number of steps to rollback
            actuator_registry: Optional actuator registry for executing rollbacks
            
        Returns:
            List[str]: List of intervention IDs that were rolled back
        """
        cursor = self.conn.cursor()
        
        # Get last N interventions (most recent first)
        cursor.execute("""
            SELECT intervention_id, lever_id, parameters_json, rollback_descriptor_json
            FROM interventions
            WHERE epoch <= ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (epoch, n_steps))
        
        interventions = cursor.fetchall()
        
        if not interventions:
            logger.warning(f"No interventions found to rollback (epoch {epoch}, n_steps {n_steps})")
            return []
        
        rolled_back_ids = []
        
        # Execute rollbacks in reverse order (oldest first)
        for intervention_id, lever_id, parameters_json, rollback_descriptor_json in reversed(interventions):
            try:
                if rollback_descriptor_json:
                    rollback_descriptor = json.loads(rollback_descriptor_json)
                    
                    # Execute rollback via actuator if available
                    if actuator_registry:
                        actuator = actuator_registry.get_actuator_for_lever(lever_id)
                        if actuator:
                            success = actuator.rollback(rollback_descriptor)
                            if success:
                                logger.info(f"Rolled back intervention {intervention_id}")
                            else:
                                logger.warning(f"Rollback failed for intervention {intervention_id}")
                        else:
                            logger.warning(f"No actuator found for lever {lever_id}")
                    else:
                        logger.info(f"Rollback descriptor for {intervention_id}: {rollback_descriptor}")
                
                rolled_back_ids.append(intervention_id)
            except Exception as e:
                logger.error(f"Error rolling back intervention {intervention_id}: {e}")
        
        # Record rollback in history
        rollback_id = str(uuid.uuid4())
        intervention_ids_json = json.dumps(rolled_back_ids)
        created_at = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO rollback_history (rollback_id, epoch, n_steps, intervention_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (rollback_id, epoch, n_steps, intervention_ids_json, created_at))
        self.conn.commit()
        
        logger.info(f"Rolled back {len(rolled_back_ids)} interventions")
        
        return rolled_back_ids
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore state from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
            
        Returns:
            Dict[str, Any]: Restored state or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT state_json FROM checkpoints WHERE checkpoint_id = ?
        """, (checkpoint_id,))
        
        row = cursor.fetchone()
        if row:
            state = json.loads(row[0])
            logger.info(f"Restored checkpoint {checkpoint_id}")
            return state
        else:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return None
    
    def get_latest_checkpoint(self, epoch: Optional[int] = None) -> Optional[str]:
        """
        Get the latest checkpoint ID.
        
        Args:
            epoch: Optional epoch to filter by (get latest before or at this epoch)
            
        Returns:
            str: Checkpoint ID or None
        """
        cursor = self.conn.cursor()
        
        if epoch is not None:
            cursor.execute("""
                SELECT checkpoint_id FROM checkpoints
                WHERE epoch <= ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (epoch,))
        else:
            cursor.execute("""
                SELECT checkpoint_id FROM checkpoints
                ORDER BY created_at DESC
                LIMIT 1
            """)
        
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_rollback_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get rollback history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List[Dict[str, Any]]: Rollback history records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rollback_id, epoch, n_steps, intervention_ids, created_at
            FROM rollback_history
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [
            {
                "rollback_id": row[0],
                "epoch": row[1],
                "n_steps": row[2],
                "intervention_ids": json.loads(row[3]),
                "created_at": row[4]
            }
            for row in rows
        ]
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up checkpoints older than retention period."""
        cutoff = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM checkpoints WHERE created_at < ?
        """, (cutoff_iso,))
        
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old checkpoints")
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

