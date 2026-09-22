from funtracks.import_export import write_to_geff
from funtracks.user_actions import UserDeleteNode

from motile_tracker.data_views.views_coordinator.tracks_list import TracksList
from motile_tracker.persistence.session import EditSession


def test_loaded_dataset_autosaves_to_its_source(solution_tracks_2d, tmp_path, qtbot):
    source = tmp_path / "original.geff"
    write_to_geff(solution_tracks_2d, source)
    widget = TracksList()
    qtbot.addWidget(widget)
    widget.add_tracks(solution_tracks_2d, "original", source_path=source)
    widget.save_dir_line.setText(str(tmp_path / "elsewhere"))
    widget.save_name_line.setText("other")
    UserDeleteNode(solution_tracks_2d, 4)
    widget.close_sessions()
    with EditSession.open(source) as session:
        assert not session.tracks.graph.has_node(4)
        session.tracks.undo()
        assert session.tracks.graph.has_node(4)
    assert not (tmp_path / "elsewhere").exists()


def test_new_results_get_unique_destinations(
    solution_tracks_2d, graph_2d, tmp_path, qtbot, monkeypatch
):
    from funtracks.data_model import SolutionTracks

    from motile_tracker.data_views.views_coordinator import tracks_list

    monkeypatch.setattr(tracks_list, "default_save_dir", lambda: tmp_path)
    widget = TracksList()
    qtbot.addWidget(widget)
    widget.add_tracks(solution_tracks_2d, "same")
    widget.add_tracks(SolutionTracks(graph_2d, ndim=3, time_attr="t"), "same")
    paths = [
        widget.tracks_list.itemWidget(
            widget.tracks_list.item(i)
        ).tracks.edit_session.path
        for i in range(2)
    ]
    widget.close_sessions()
    assert paths[0] != paths[1]
    assert all(p.parent == tmp_path and p.exists() for p in paths)


def test_load_recovers_history_before_reading_geff(solution_tracks_2d, tmp_path, qtbot):
    import shutil

    source = tmp_path / "source.geff"
    with EditSession.create(solution_tracks_2d, source):
        UserDeleteNode(solution_tracks_2d, 4)
    shutil.rmtree(source / "nodes")
    widget = TracksList()
    qtbot.addWidget(widget)
    widget.file_dialog.exec_ = lambda: True
    widget.file_dialog.selectedFiles = lambda: [str(source)]
    tracks, name, path = widget.load_internal_tracks()
    widget.add_tracks(tracks, name, source_path=path)
    assert not tracks.graph.has_node(4)
    tracks.undo()
    assert tracks.graph.has_node(4)
    widget.close_sessions()


def test_motile_loader_recovers_missing_graph(graph_2d, tmp_path, qtbot):
    import shutil

    from motile_tracker.motile.backend.motile_run import MotileRun
    from motile_tracker.motile.backend.solver_params import SolverParams

    tracks = MotileRun(graph_2d, "run", solver_params=SolverParams())
    path = tmp_path / "run.geff"
    with EditSession.create(tracks, path):
        UserDeleteNode(tracks, 4)
    shutil.rmtree(path / "nodes")
    widget = TracksList()
    qtbot.addWidget(widget)
    widget.file_dialog.exec_ = lambda: True
    widget.file_dialog.selectedFiles = lambda: [str(path)]
    loaded, _, _ = widget.load_motile_run()
    assert isinstance(loaded, MotileRun)
    assert not loaded.graph.has_node(4)
    loaded.undo()
    assert loaded.graph.has_node(4)
    widget.close_sessions()


def test_readonly_geff_uses_writable_working_copy(solution_tracks_2d, tmp_path, qtbot):
    import os

    path = tmp_path / "readonly.geff"
    with EditSession.create(solution_tracks_2d, path):
        UserDeleteNode(solution_tracks_2d, 4)
    files = list(path.rglob("*")) + [path]
    try:
        for file in files:
            file.chmod(0o555 if file.is_dir() else 0o444)
        widget = TracksList()
        qtbot.addWidget(widget)
        widget.file_dialog.exec_ = lambda: True
        widget.file_dialog.selectedFiles = lambda: [str(path)]
        tracks, name, source = widget.load_internal_tracks()
        assert source != path
        assert tracks.edit_session.path == source
        tracks.undo()
        assert tracks.graph.has_node(4)
        widget.close_sessions()
    finally:
        for file in files:
            os.chmod(file, 0o755 if file.is_dir() else 0o644)
    with EditSession.open(path) as unchanged:
        assert not unchanged.tracks.graph.has_node(4)
