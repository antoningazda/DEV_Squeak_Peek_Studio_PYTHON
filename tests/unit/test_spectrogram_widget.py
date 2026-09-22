"""Tests for SpectrogramWidget colormap, label colors, and interactive features."""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from squeak_peek.gui._spectrogram_widget import (
    SpectrogramWidget,
    _build_colormap,
    _get_color_for_name,
    _PARULA_LUT,
    _TURBO_LUT,
)
from squeak_peek.labels.model import Label


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication once per test session."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    """Create a fresh SpectrogramWidget for each test."""
    w = SpectrogramWidget()
    yield w
    w.deleteLater()


class TestColorFunction:
    """Test _get_color_for_name() helper."""

    def test_valid_colors(self):
        """Test all 7 supported color names."""
        colors = {
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
            "cyan": (0.0, 1.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0, 1.0),
            "yellow": (1.0, 1.0, 0.0, 1.0),
            "white": (1.0, 1.0, 1.0, 1.0),
        }
        for name, expected in colors.items():
            assert _get_color_for_name(name) == expected
            # Case-insensitive
            assert _get_color_for_name(name.upper()) == expected

    def test_none_color(self):
        """None should return None."""
        assert _get_color_for_name(None) is None

    def test_invalid_color(self):
        """Invalid color name should return None."""
        assert _get_color_for_name("invalid") is None


class TestColormapBuilder:
    """Test _build_colormap() function."""

    def test_parula_colormap(self):
        """Parula should build successfully."""
        cm = _build_colormap("parula")
        assert cm is not None
        # Verify it's a pyqtgraph ColorMap with correct structure
        assert hasattr(cm, "pos")
        assert hasattr(cm, "color")

    def test_turbo_colormap(self):
        """Turbo should build successfully."""
        cm = _build_colormap("turbo")
        assert cm is not None
        assert hasattr(cm, "pos")
        assert hasattr(cm, "color")

    def test_invgray_colormap(self):
        """Invgray (reversed gray) should build successfully."""
        cm = _build_colormap("invgray")
        assert cm is not None

    def test_matplotlib_colormaps(self):
        """Standard matplotlib colormap names should work."""
        names = ["hsv", "hot", "cool", "spring", "summer", "autumn", "winter",
                 "gray", "bone", "copper", "pink", "jet"]
        for name in names:
            cm = _build_colormap(name)
            assert cm is not None

    def test_case_insensitivity(self):
        """Colormap names should be case-insensitive."""
        cm1 = _build_colormap("PARULA")
        cm2 = _build_colormap("parula")
        assert cm1 is not None and cm2 is not None

    def test_invalid_colormap(self):
        """Invalid colormap name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown colormap"):
            _build_colormap("nonexistent")

    def test_parula_lut_shape(self):
        """Parula LUT should be 256x3."""
        assert _PARULA_LUT.shape == (256, 3)
        assert _PARULA_LUT.dtype == np.float32

    def test_turbo_lut_shape(self):
        """Turbo LUT should be 256x3."""
        assert _TURBO_LUT.shape == (256, 3)
        assert _TURBO_LUT.dtype == np.float32


class TestSpectrogramWidgetColormaps:
    """Test SpectrogramWidget colormap functionality."""

    def test_set_colormap(self, widget):
        """set_colormap() should update the colormap."""
        widget.set_colormap("parula")
        assert widget._current_colormap_name == "parula"

        widget.set_colormap("jet")
        assert widget._current_colormap_name == "jet"

    def test_display_with_colormap_kwarg(self, widget):
        """display() should accept colormap_name kwarg."""
        # Create simple test data
        samples = np.random.randn(10000)
        fs = 192000

        # Should not raise
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            colormap_name="turbo",
        )
        assert widget._current_colormap_name == "turbo"

        # Change colormap via display
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            colormap_name="hsv",
        )
        assert widget._current_colormap_name == "hsv"


class TestSpectrogramWidgetLabelColors:
    """Test SpectrogramWidget label color functionality."""

    def test_display_with_color_kwargs(self, widget):
        """display() should accept detected_color_name and reference_color_name."""
        samples = np.random.randn(10000)
        fs = 192000
        detected = [Label(start_time=0.01, end_time=0.02, label="test")]
        reference = [Label(start_time=0.015, end_time=0.025, label="ref")]

        # Should not raise
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            detected_labels=detected,
            reference_labels=reference,
            detected_color_name="red",
            reference_color_name="blue",
        )

    def test_color_override_none_fallback(self, widget):
        """None or omitted color kwargs should use theme colors."""
        samples = np.random.randn(10000)
        fs = 192000
        detected = [Label(start_time=0.01, end_time=0.02, label="test")]

        # Should use theme colors
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            detected_labels=detected,
            detected_color_name=None,  # explicitly None
        )

        # Should also use theme colors
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            detected_labels=detected,
            # detected_color_name omitted
        )


class TestBoundaryDrag:
    """Test boundary drag functionality."""

    def test_enable_boundary_drag(self, widget):
        """enable_boundary_drag() should add two draggable lines."""
        widget.enable_boundary_drag(0.01, 0.02, 40, 120)

        assert "start" in widget._boundary_lines
        assert "end" in widget._boundary_lines

    def test_disable_boundary_drag(self, widget):
        """disable_boundary_drag() should remove lines."""
        widget.enable_boundary_drag(0.01, 0.02, 40, 120)
        assert len(widget._boundary_lines) == 2

        widget.disable_boundary_drag()
        assert len(widget._boundary_lines) == 0

    def test_enable_idempotent(self, widget):
        """Calling enable_boundary_drag() again should replace lines."""
        widget.enable_boundary_drag(0.01, 0.02, 40, 120)
        first_lines = dict(widget._boundary_lines)

        widget.enable_boundary_drag(0.01, 0.03, 40, 120)
        second_lines = dict(widget._boundary_lines)

        # Should still have 2 lines but they're different objects
        assert len(second_lines) == 2
        assert second_lines["start"] is not first_lines["start"]

    def test_boundary_dragged_signal(self, widget):
        """boundary_dragged signal should be emitted."""
        widget.enable_boundary_drag(0.01, 0.02, 40, 120)

        received = []
        widget.boundary_dragged.connect(lambda edge, t: received.append((edge, t)))

        # Manually trigger the signal emission
        widget._on_boundary_line_moved("start", widget._boundary_lines["start"])

        assert received
        assert received[0][0] == "start"


class TestRightClick:
    """Test right-click functionality."""

    def test_spectrogram_right_clicked_signal_exists(self, widget):
        """spectrogram_right_clicked signal should exist."""
        assert hasattr(widget, "spectrogram_right_clicked")

    def test_mouse_click_handler_registered(self, widget):
        """Mouse click handlers should be connected."""
        # Just verify no exceptions during initialization
        assert widget._spec_plot.scene() is not None
        assert widget._wave_plot.scene() is not None


class TestDisplayBackwardCompatibility:
    """Ensure new kwargs are optional and backward-compatible."""

    def test_display_without_new_kwargs(self, widget):
        """Old code using display() without new kwargs should still work."""
        samples = np.random.randn(10000)
        fs = 192000

        # Call with only required and original optional args
        widget.display(
            samples,
            fs,
            0.0,
            0.05,
            fmin_hz=40_000,
            fmax_hz=120_000,
            detected_labels=None,
            reference_labels=None,
            show_detected=True,
            show_reference=True,
        )
        # Should not raise

    def test_display_minimal(self, widget):
        """Minimal call with only required args."""
        samples = np.random.randn(10000)
        fs = 192000

        widget.display(samples, fs, 0.0, 0.05)
        # Should not raise
