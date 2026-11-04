"""Tests for main_app.py: StartupWidget, MainAppWidget, and single menu widget classes."""

from qtpy.QtWidgets import QWidget

from mm_track.application_menus.main_app import (
    MENU_WIDGETS,
    EditingGroupWidget,
    EditingSelection_LauncherWidget,
    GettingStarted_LauncherWidget,
    Groups_LauncherWidget,
    LineageView_LauncherWidget,
    MainAppWidget,
    StartupWidget,
    Table_LauncherWidget,
    Tracking_LauncherWidget,
    TrackingGroupWidget,
    TrackList_LauncherWidget,
    Visualization_LauncherWidget,
)


def test_startupwidget_initializes_all_tabs(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = StartupWidget(viewer)
    assert isinstance(widget, QWidget)
    # Should initialize all menu widgets
    assert len(viewer.window.dock_widgets) == len(MENU_WIDGETS)


def test_main_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = MainAppWidget(viewer)
    assert isinstance(widget, StartupWidget)
    assert len(viewer.window.dock_widgets) == len(MENU_WIDGETS)


def test_single_menu_widgets(qtbot, make_napari_viewer):
    # Each single menu widget should only initialize one docked widget
    for WidgetClass in [
        GettingStarted_LauncherWidget,
        Tracking_LauncherWidget,
        TrackList_LauncherWidget,
        EditingSelection_LauncherWidget,
        Visualization_LauncherWidget,
        Groups_LauncherWidget,
        Table_LauncherWidget,
        LineageView_LauncherWidget,
    ]:
        n_docked_widgets = 0
        viewer = make_napari_viewer()
        widget = WidgetClass(viewer)
        n_docked_widgets += 1
        assert isinstance(widget, StartupWidget)
        assert len(viewer.window.dock_widgets) == n_docked_widgets


def test_tracking_and_editing_group_widgets(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    tracking = TrackingGroupWidget(viewer)
    editing = EditingGroupWidget(viewer)
    assert isinstance(tracking, StartupWidget)
    assert isinstance(editing, StartupWidget)
    # Should initialize subset of widgets: the 4 tracking widgets plus the editing
    # widgets not already among them (Editing && Selection, Kymograph Data, Features,
    # Groups)
    assert len(viewer.window.dock_widgets) == 8


def test_default_layout(qtbot, make_napari_viewer):
    """The full app puts the lineage tree beside napari's layer controls and splits a
    strip along the bottom between a compact Editing & Selection and the other menus,
    leaving the right side of the window to the canvas."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QDockWidget, QTabBar

    from mm_track.application_menus.editing_selection_menu import (
        EditingSelectionWidget,
    )

    viewer = make_napari_viewer()
    MainAppWidget(viewer)
    QApplication.processEvents()  # the layout is made once the docks are in place
    window = viewer.window._qt_window
    docks = {d.windowTitle(): d for d in window.findChildren(QDockWidget)}
    area = {name: window.dockWidgetArea(d) for name, d in docks.items()}

    assert area["Lineage View"] == Qt.LeftDockWidgetArea
    assert window.tabifiedDockWidgets(docks["Lineage View"]) == []
    for name in MENU_WIDGETS:
        if name != "Lineage View":
            assert area[name] == Qt.BottomDockWidgetArea, name
    assert window.tabifiedDockWidgets(docks["Editing && Selection"]) == []
    assert Qt.RightDockWidgetArea not in area.values()
    # the menus beside it open on Kymograph Data
    tabbars = window.findChildren(QTabBar)
    assert "Kymograph Data" in [bar.tabText(bar.currentIndex()) for bar in tabbars]

    editing = window.findChildren(EditingSelectionWidget)[0]
    assert editing.editing_widget.set_division_btn.text() == "Division [Y]"
    assert editing.editing_widget.isHidden()  # the tall layout is replaced


def test_compact_layout_shows_every_button(qtbot, make_napari_viewer):
    from qtpy.QtWidgets import QPushButton

    from mm_track.application_menus.editing_selection_menu import (
        EditingSelectionWidget,
    )

    viewer = make_napari_viewer()
    widget = EditingSelectionWidget(viewer)
    buttons = widget.findChildren(QPushButton)
    widget.use_compact_layout()
    widget.show()
    qtbot.addWidget(widget)

    assert set(widget.findChildren(QPushButton)) == set(buttons)  # the same, moved
    assert all(b.isVisibleTo(widget) for b in buttons)
    widget.editing_widget.update_track_id_color()
    assert widget.editing_widget.label.text().startswith("Track ID:")
