import pytest

from mm_track.import_export import geff_collection
from mm_track.persistence.session import EditSession


def test_failed_image_loading_releases_session(geff_collection_store, monkeypatch):
    path = geff_collection_store / "trench_a" / "tracking_graph.geff"
    loaded = geff_collection.load_geff_group(path)
    with EditSession.create(loaded.tracks, path):
        pass

    def fail(path):
        raise ValueError("broken image")

    monkeypatch.setattr(geff_collection, "open_array", fail)
    with pytest.raises(ValueError, match="broken image"):
        geff_collection.load_geff_group(path)
    with EditSession.open(path) as session:
        assert session.tracks.graph_solution.num_nodes() > 0


def test_projection_keeps_the_label_array_in_step_with_the_edits(
    geff_collection_store,
):
    """An edit is published in the dataset's own format: the label array the GEFF
    refers to is rewritten from the edited masks, and the GEFF still refers to it."""
    import numpy as np
    import zarr
    from funtracks.user_actions import UserDeleteNode

    path = geff_collection_store / "trench_a" / "tracking_graph.geff"
    loaded = geff_collection.load_geff_group(path)
    related = geff_collection.read_related_arrays(path)
    original = zarr.open_array(related.labels_path, mode="r")[:]
    node = loaded.tracks.graph_solution.node_ids()[-1]
    time = loaded.tracks.get_time(node)
    label = int(loaded.tracks.graph_solution.nodes[node]["seg_id"])
    with EditSession.create(loaded.tracks, path) as session:
        UserDeleteNode(session.tracks, node)
        session.flush()
    current = geff_collection.read_related_arrays(path)
    assert current.image_path == related.image_path
    assert current.labels_path == related.labels_path
    edited = zarr.open_array(related.labels_path, mode="r")[:]
    assert not np.any(edited[time] == label)
    # everything else in the array is as it was
    original[time][original[time] == label] = 0
    np.testing.assert_array_equal(edited, original)
