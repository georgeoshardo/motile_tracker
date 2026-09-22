import pytest

from motile_tracker.import_export import geff_collection
from motile_tracker.persistence.session import EditSession


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


def test_projection_uses_edited_masks_not_original_label_reference(
    geff_collection_store,
):
    import numpy as np
    import zarr
    from funtracks.import_export import import_from_geff
    from funtracks.user_actions import UserDeleteNode

    path = geff_collection_store / "trench_a" / "tracking_graph.geff"
    loaded = geff_collection.load_geff_group(path)
    related = geff_collection.read_related_arrays(path)
    original = zarr.open_array(related.labels_path, mode="r")[:]
    node = loaded.tracks.graph_solution.node_ids()[-1]
    time = loaded.tracks.get_time(node)
    with EditSession.create(loaded.tracks, path) as session:
        UserDeleteNode(session.tracks, node)
        session.flush()
    current = geff_collection.read_related_arrays(path)
    assert current.image_path == related.image_path
    assert current.labels_path is None
    np.testing.assert_array_equal(
        zarr.open_array(related.labels_path, mode="r")[:], original
    )
    result = import_from_geff(path)
    assert not result.graph_solution.has_node(node)
    assert not np.any(np.asarray(result.segmentation[time]) == node)
