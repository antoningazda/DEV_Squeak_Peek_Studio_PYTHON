"""
Label data model.

A Label represents a single detected or annotated USV event.
It mirrors the MATLAB struct:

    labels(i).StartTime       % float, seconds
    labels(i).EndTime         % float, seconds
    labels(i).Label           % string
    labels(i).StartFrequency  % float, Hz
    labels(i).EndFrequency    % float, Hz
    labels(i).StartIndex      % int, sample index
    labels(i).StopIndex       % int, sample index

Phase 4 status: model defined here in Phase 1 so other modules can
import it for type hints. I/O and post-processing come in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Label:
    """A single USV event annotation."""

    start_time: float        # seconds
    end_time: float          # seconds
    label: str = ""          # call type, e.g. "d", "5", "sk"
    start_frequency: float = 0.0   # Hz (0 when unused by detector)
    end_frequency: float   = 0.0   # Hz
    start_index: int = 0     # sample index
    stop_index: int  = 0     # sample index
    detection_state: str = "None"  # "None", "Accepted", "Rejected"
    classification_state: str = "None"  # "None", "Accepted", "Rejected"

    @property
    def duration(self) -> float:
        """Duration of the event in seconds."""
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        """Midpoint time in seconds (used by the midpoint matching criterion)."""
        return (self.start_time + self.end_time) / 2.0

    def __repr__(self) -> str:
        return (
            f"Label(start={self.start_time:.4f}s, end={self.end_time:.4f}s, "
            f"label='{self.label}', dur={self.duration*1000:.1f}ms)"
        )
