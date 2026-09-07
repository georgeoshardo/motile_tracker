import argparse
import sys
from pathlib import Path

import napari
from qtpy.QtCore import QTimer

from motile_tracker.application_menus.main_app import StartupWidget


def load_startup_data(
    viewer: napari.Viewer, store: Path, group: str | None, kymograph: bool
) -> None:
    """Load a GEFF group from `store` into the viewer, through the Kymograph Data
    widget when it is present (so its store path and group list are filled in),
    or directly otherwise.
    """
    from motile_tracker.application_menus.kymograph_data_widget import (
        KymographDataWidget,
    )
    from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
    from motile_tracker.import_export.geff_collection import (
        add_geff_group_to_viewer,
        find_geff_groups,
    )

    tracks_viewer = TracksViewer.get_instance(viewer)
    widget = None
    menu_manager = tracks_viewer.menu_manager
    if menu_manager is not None:
        wrapper = menu_manager.menu_widgets.get("Kymograph Data")
        if wrapper is not None and isinstance(wrapper.widget(), KymographDataWidget):
            widget = wrapper.widget()

    if widget is not None:
        groups = widget.set_store_path(store)
        if not groups:
            return
        if group is not None and not widget.select_group(group):
            print(f"No GEFF group named {group!r} in {store}", file=sys.stderr)
            return
        widget.kymograph_checkbox.setChecked(kymograph)
        widget.load_selected()
        return

    groups = find_geff_groups(store)
    if not groups:
        print(f"No GEFF groups found in {store}", file=sys.stderr)
        return
    selected = groups[0]
    if group is not None:
        matches = [g for g in groups if g.name == group]
        if not matches:
            print(f"No GEFF group named {group!r} in {store}", file=sys.stderr)
            return
        selected = matches[0]
    add_geff_group_to_viewer(viewer, selected.path, selected.name, kymograph=kymograph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "tracking", "editing"],
        default="all",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Zarr store holding GEFF groups with related images/segmentations "
        "(e.g. one group per mother machine trench) to load at startup.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Name of the GEFF group inside --data to load (default: the first).",
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Show the data from --data in the spatial view instead of the kymograph.",
    )

    args, _ = parser.parse_known_args()

    viewer = napari.Viewer()
    StartupWidget(viewer, mode=args.mode)

    if args.data is not None:
        # after StartupWidget has docked its widgets (it finishes via QTimer too)
        QTimer.singleShot(
            0,
            lambda: load_startup_data(
                viewer, args.data, args.group, kymograph=not args.spatial
            ),
        )

    napari.run()


if __name__ == "__main__":
    sys.exit(main())
