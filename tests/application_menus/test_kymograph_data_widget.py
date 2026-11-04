from pathlib import Path

from mm_track.application_menus.kymograph_data_widget import KymographDataWidget
from mm_track.application_menus.main_app import MENU_WIDGETS
from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer


def test_widget_is_registered_in_the_menus():
    assert MENU_WIDGETS["Kymograph Data"]["widget"] is KymographDataWidget


def test_scan_lists_groups(make_napari_viewer, geff_collection_store, qtbot):
    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)

    groups = widget.set_store_path(geff_collection_store)

    assert [group.name for group in groups] == ["trench_a", "trench_b"]
    assert [widget.group_box.itemText(i) for i in range(2)] == ["trench_a", "trench_b"]
    assert "2 GEFF groups" in widget.status_label.text()
    assert not widget.previous_button.isEnabled()
    assert widget.next_button.isEnabled()


def test_scan_reports_problems(make_napari_viewer, tmp_path, qtbot):
    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)

    widget.set_store_path(tmp_path / "does_not_exist")
    assert "Not a directory" in widget.status_label.text()

    widget.set_store_path(tmp_path)
    assert not widget.next_button.isEnabled()
    assert "No GEFF groups" in widget.status_label.text()

    widget.path_edit.setText("")
    assert widget.scan() == []
    assert widget.status_label.text() == ""


def test_picking_a_group_loads_it(make_napari_viewer, geff_collection_store, qtbot):
    """Listing the groups loads nothing; picking one from the list loads it, and
    picking the one already loaded does not load it again."""
    from qtpy.QtCore import Qt

    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)
    loads = []
    load_selected = widget.load_selected
    widget.load_selected = lambda: loads.append(load_selected()) or loads[-1]
    widget.set_store_path(geff_collection_store)
    assert loads == []

    qtbot.keyClick(widget.group_box, Qt.Key_Down)  # pick trench_b

    assert [loaded.name for loaded in loads] == ["trench_b"]
    assert "Loaded trench_b" in widget.status_label.text()

    widget.group_box.activated.emit(1)  # trench_b again
    assert len(loads) == 1


def test_load_next_and_previous(make_napari_viewer, geff_collection_store, qtbot):
    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)
    widget.set_store_path(geff_collection_store)
    tracks_viewer = TracksViewer.get_instance(viewer)

    loaded_a = widget.load_selected()

    assert loaded_a.name == "trench_a"
    assert tracks_viewer.view_mode == "kymograph"
    assert tracks_viewer.kymograph_layers.detached_image_layer is loaded_a.image_layer
    assert "Loaded trench_a: 6 nodes, 4 edges" in widget.status_label.text()

    loaded_b = widget.load_next()

    assert loaded_b.name == "trench_b"
    assert widget.group_box.currentText() == "trench_b"
    assert tracks_viewer.view_mode == "kymograph"
    assert tracks_viewer.kymograph_layers.detached_image_layer is loaded_b.image_layer
    # the earlier image was removed, the earlier tracks are still listed
    assert loaded_a.image_layer not in viewer.layers
    assert tracks_viewer.tracks_list.tracks_list.count() == 2
    assert widget.load_next() is None  # no further group

    tracks_viewer.set_view_mode("spatial")
    assert loaded_a.image_layer not in viewer.layers
    assert loaded_b.image_layer in viewer.layers

    loaded_a_again = widget.load_previous()
    assert loaded_a_again.name == "trench_a"
    assert widget.load_previous() is None


def test_keep_previous_images_and_spatial_view(
    make_napari_viewer, geff_collection_store, qtbot
):
    viewer = make_napari_viewer()
    widget = KymographDataWidget(viewer)
    qtbot.addWidget(widget)
    widget.set_store_path(geff_collection_store)
    widget.kymograph_checkbox.setChecked(False)
    widget.replace_checkbox.setChecked(False)
    tracks_viewer = TracksViewer.get_instance(viewer)

    loaded_a = widget.load_selected()
    loaded_b = widget.load_next()

    assert tracks_viewer.view_mode == "spatial"
    assert loaded_a.image_layer in viewer.layers
    assert loaded_b.image_layer in viewer.layers
    assert widget.select_group("trench_a") is True
    assert widget.select_group("nope") is False
    assert widget.current_group().path == Path(loaded_a.geff_path)
