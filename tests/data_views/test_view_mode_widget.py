from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mm_track.application_menus.view_mode_widget import ViewModeWidget
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    """Clear viewer layers between tests."""
    yield
    viewer.layers.clear()


@pytest.fixture
def view_mode_widget(viewer, solution_tracks_3d, qtbot):
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_3d, name="test")

    widget = ViewModeWidget(viewer)
    qtbot.addWidget(widget)

    return widget, tracks_viewer


@pytest.fixture
def view_mode_widget_2d(viewer, solution_tracks_2d, segmentation_2d, qtbot):
    """A ViewModeWidget on 2D+time tracks with a matching raw image layer, so that the
    kymograph view is available."""
    viewer.add_image(np.asarray(segmentation_2d, dtype=float), name="raw")

    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")
    tracks_viewer.set_kymograph_image_layer("raw")

    widget = ViewModeWidget(viewer)
    qtbot.addWidget(widget)

    return widget, tracks_viewer


@pytest.mark.parametrize("mode", ["lineage", "group", "all"])
def test_switch_display_modes(view_mode_widget, mode):
    """Test that checking the radio buttons changes the display mode"""

    widget, tracks_viewer = view_mode_widget

    widget.mode_widget.button_for_mode(mode).setChecked(True)

    assert tracks_viewer.mode == mode


@pytest.mark.parametrize("mode", ["lineage", "group", "all"])
def test_mode_update_syncs_radio_buttons(view_mode_widget, mode):
    """Check that the radio button states are updated when the tracks_viewer updates its
    display mode."""

    widget, tracks_viewer = view_mode_widget

    tracks_viewer.set_display_mode(mode)
    widget._update_controls()
    radio = widget.mode_widget.button_for_mode(mode)

    assert radio.isChecked()


# Ortho-views integration tests
class TestOrthoViewsIntegration:
    """Tests for orthogonal views checkbox and initialization."""

    def test_ortho_views_checkbox_initially_unchecked(self, view_mode_widget):
        """Test that the ortho views checkbox starts unchecked."""
        widget, _ = view_mode_widget
        assert not widget.show_ortho_views.isChecked()

    @patch(
        "mm_track.application_menus.view_mode_widget._viewer_managers",
        new=dict,
    )
    @patch("mm_track.application_menus.view_mode_widget.initialize_ortho_views")
    def test_initialize_ortho_views_viewer_not_in_managers(
        self, mock_init, view_mode_widget
    ):
        """Test ortho views initialization when viewer is not already in _VIEWER_MANAGERS."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_manager.main_controls_widget.show_orth_views.connect = MagicMock(
            return_value=MagicMock()
        )
        mock_init.return_value = mock_manager

        # Trigger checkbox
        widget.show_ortho_views.setChecked(True)

        # verify initialize_ortho_views was called
        mock_init.assert_called_once_with(widget.viewer)

        # verify manager was stored
        assert widget.orth_view_manager is not None
        assert widget.orth_views_connection is not None

    def test_initialize_ortho_views_with_existing_manager(self, view_mode_widget):
        """Test ortho views when viewer is already in _VIEWER_MANAGERS."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_manager.show = MagicMock()
        mock_manager.hide = MagicMock()

        # Mock the _VIEWER_MANAGERS to already contain this viewer
        with patch(
            "mm_track.application_menus.view_mode_widget._viewer_managers",
            return_value={widget.viewer: mock_manager},
        ):
            widget.show_ortho_views.setChecked(True)

            # verify manager.show() was called
            mock_manager.show.assert_called_once()

    def test_ortho_views_hide_when_unchecked(self, view_mode_widget):
        """Test that ortho views are hidden and resized when checkbox is unchecked."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_manager.show = MagicMock()
        mock_manager.hide = MagicMock()
        mock_manager.set_splitter_sizes = MagicMock()

        with patch(
            "mm_track.application_menus.view_mode_widget._viewer_managers",
            return_value={widget.viewer: mock_manager},
        ):
            # Check the box first
            widget.show_ortho_views.setChecked(True)
            mock_manager.show.assert_called_once()

            # Uncheck the box
            widget.show_ortho_views.setChecked(False)

            # Verify hide and set_splitter_sizes were called
            mock_manager.hide.assert_called_once()
            mock_manager.set_splitter_sizes.assert_called_once_with(0.0, 0.0)

    @patch("mm_track.application_menus.view_mode_widget.initialize_ortho_views")
    def test_ortho_views_signal_connection(self, mock_init, view_mode_widget):
        """Test that the ortho view manager's signal is connected to the widget."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_signal = MagicMock()
        mock_manager.main_controls_widget.show_orth_views = mock_signal
        mock_manager.main_controls_widget.destroyed = MagicMock()
        mock_signal.connect = MagicMock(return_value=MagicMock())

        mock_init.return_value = mock_manager

        widget.show_ortho_views.setChecked(True)

        # Verify signal was connected to initialize_ortho_views
        mock_signal.connect.assert_called_once()
        call_args = mock_signal.connect.call_args[0]
        assert call_args[0] == widget.initialize_ortho_views

    @patch("mm_track.application_menus.view_mode_widget.initialize_ortho_views")
    def test_on_ortho_cleanup(self, mock_init, view_mode_widget):
        """Test cleanup when ortho view manager is destroyed."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_manager.main_controls_widget.show_orth_views = MagicMock()
        mock_manager.main_controls_widget.destroyed = MagicMock()
        mock_manager.main_controls_widget.show_orth_views.connect = MagicMock(
            return_value=MagicMock()
        )

        mock_init.return_value = mock_manager

        widget.show_ortho_views.setChecked(True)

        # Simulate widget destruction
        widget._on_ortho_cleanup()

        # Verify checkbox is unchecked and disconnected
        assert not widget.show_ortho_views.isChecked()
        assert widget.orth_view_manager is None
        assert widget.orth_views_connection is None

    @patch("mm_track.application_menus.view_mode_widget.initialize_ortho_views")
    def test_disconnect_ortho_views_with_valid_connection(
        self, mock_init, view_mode_widget
    ):
        """Test _disconnect_ortho_views with valid connection."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_signal = MagicMock()
        mock_manager.main_controls_widget.show_orth_views = mock_signal
        mock_manager.main_controls_widget.destroyed = MagicMock()
        mock_signal.connect = MagicMock(return_value=MagicMock())
        mock_signal.disconnect = MagicMock()

        mock_init.return_value = mock_manager

        widget.show_ortho_views.setChecked(True)

        # Disconnect
        widget._disconnect_ortho_views()

        # Verify disconnect was called and connection is cleared
        mock_signal.disconnect.assert_called_once()
        assert widget.orth_views_connection is None
        assert widget.orth_view_manager is None

        # No-op, should not raise
        widget._disconnect_ortho_views()

        assert widget.orth_views_connection is None
        assert widget.orth_view_manager is None

    def test_initialize_ortho_views_syncs_checkbox_state(self, view_mode_widget):
        """Test that initialize_ortho_views syncs checkbox state."""
        widget, _ = view_mode_widget

        mock_manager = MagicMock()
        mock_manager.main_controls_widget.show_orth_views = MagicMock()
        mock_manager.main_controls_widget.destroyed = MagicMock()
        mock_manager.main_controls_widget.show_orth_views.connect = MagicMock(
            return_value=MagicMock()
        )

        with patch(
            "mm_track.application_menus.view_mode_widget._viewer_managers",
            return_value={widget.viewer: mock_manager},
        ):
            # Manually call with checked=False (simulating external unchecking)
            widget.initialize_ortho_views(False)

            # Verify checkbox state is synced
            assert not widget.show_ortho_views.isChecked()


class TestKymographView:
    """The view mode buttons and the kymograph image selector."""

    def test_kymograph_button_disabled_for_3d_tracks(self, view_mode_widget):
        """The kymograph view only supports 2D+time tracks."""
        widget, _ = view_mode_widget

        assert widget.view_mode_widget.button_for_mode("kymograph").isEnabled() is False

    def test_view_mode_switches_to_kymograph_and_disables_image_selector(
        self, view_mode_widget_2d
    ):
        widget, tracks_viewer = view_mode_widget_2d

        assert widget.view_mode_widget.button_for_mode("kymograph").isEnabled() is True
        assert widget.image_layer_box.currentText() == "raw"

        widget.view_mode_widget.button_for_mode("kymograph").setChecked(True)

        assert tracks_viewer.view_mode == "kymograph"
        # the image can only be changed from the spatial view
        assert widget.image_layer_box.isEnabled() is False
        assert widget.image_layer_box.currentText() == "raw"

        widget.view_mode_widget.button_for_mode("spatial").setChecked(True)

        assert tracks_viewer.view_mode == "spatial"
        assert widget.image_layer_box.isEnabled() is True
