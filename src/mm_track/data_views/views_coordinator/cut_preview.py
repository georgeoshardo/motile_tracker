"""The cuts that Shift+C proposes, shown before they are applied.

For every selected node that an alternative segmentation cuts (see
:mod:`hypothesis_cut`), the pieces of its current candidate are drawn as outlines in
a temporary labels layer over its mask, in the spatial view or at its frame of the
kymograph page, and a message says what is proposed. Shift+C steps every node that
has more than one candidate to its next one, Enter applies them all
(:meth:`TracksViewer.apply_cut_preview`), Esc, another selection, any edit or a
change of view clears them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from napari.utils import DirectLabelColormap
from napari.utils.notifications import show_info

from mm_track.data_views.views_coordinator.hypothesis_cut import Cut

if TYPE_CHECKING:
    import napari

    from mm_track.data_views.views_coordinator.tracks_viewer import TracksViewer

LAYER_NAME = "cut preview"
# outline colours of the pieces, in piece order
COLOURS = ["magenta", "cyan", "yellow", "lime", "orange", "white"]
# labels per node in the preview layer, so neighbouring nodes' pieces stay apart
SLOTS = 16


def describe(names: tuple[str, ...], shown: int = 3) -> str:
    """'e350, e950, e994' or 'e650, e750 +6'."""
    if len(names) <= shown:
        return ", ".join(names)
    return f"{', '.join(names[: shown - 1])} +{len(names) - shown + 1}"


@dataclass
class Proposal:
    """The candidate cuts of one node and the one shown.

    Attributes:
        node: The node to cut.
        time: Its frame.
        cuts: The candidates, best first.
        uncut: Continuity of its mask left whole (for comparison).
        index: The candidate shown.
    """

    node: int
    time: int
    cuts: list[Cut]
    uncut: float
    index: int = 0

    @property
    def current(self) -> Cut:
        return self.cuts[self.index]


class CutPreview:
    """The proposed cuts of the selected nodes, and the layer showing them.

    Attributes:
        proposals: One per node that a hypothesis cuts, earliest frame first.
        skipped: Selected nodes that no hypothesis cuts.
        looked_at: How many hypotheses were looked at (per frame, at most).
    """

    def __init__(self, tracks_viewer: TracksViewer):
        self.tracks_viewer = tracks_viewer
        self.layer: napari.layers.Labels | None = None
        self.proposals: list[Proposal] = []
        self.skipped: list[int] = []
        self.looked_at = 0

    @property
    def active(self) -> bool:
        return bool(self.proposals)

    @property
    def selection(self) -> set[int]:
        """The selection the proposals were made for."""
        return {p.node for p in self.proposals} | set(self.skipped)

    # the one-node view of the proposals, for a single selected node
    @property
    def node(self) -> int | None:
        return self.proposals[0].node if self.active else None

    @property
    def current(self) -> Cut | None:
        return self.proposals[0].current if self.active else None

    @property
    def index(self) -> int:
        return self.proposals[0].index if self.active else 0

    @property
    def cuts(self) -> list[Cut]:
        return self.proposals[0].cuts if self.active else []

    def show(self, proposals: list[Proposal], skipped: list[int], looked_at: int):
        self.proposals = sorted(proposals, key=lambda p: p.time)
        self.skipped, self.looked_at = list(skipped), int(looked_at)
        self.draw()
        self.announce()

    def step(self):
        """Show each node's next candidate (after its last, its first again)."""
        if not self.active:
            return
        for proposal in self.proposals:
            proposal.index = (proposal.index + 1) % len(proposal.cuts)
        self.draw()
        self.announce()

    def clear(self):
        self.proposals, self.skipped = [], []
        viewer = self.tracks_viewer.viewer
        if self.layer is not None and self.layer in viewer.layers:
            viewer.layers.remove(self.layer)
        self.layer = None

    def _fit(self, proposal: Proposal) -> str:
        cut = proposal.current
        frame = {"before": f"t{proposal.time - 1}", "after": f"t{proposal.time + 1}"}.get(cut.frame)
        if not frame:
            return "no neighbouring frame to compare with"
        return f"fits {frame} {cut.continuity:.2f} (uncut {proposal.uncut:.2f})"

    @staticmethod
    def _worse(proposal: Proposal) -> bool:
        cut = proposal.current
        return bool(cut.frame) and cut.continuity < proposal.uncut

    def message(self) -> str:
        if not self.active:
            return ""
        if len(self.proposals) == 1 and not self.skipped:
            p = self.proposals[0]
            cut = p.current
            worse = (
                " Fits the neighbouring frames worse than leaving the mask whole."
                if self._worse(p)
                else ""
            )
            return (
                f"Cut {p.index + 1} of {len(p.cuts)} for node {p.node}: "
                f"{len(cut.pieces)} pieces from {describe(cut.hypotheses)} "
                f"({len(cut.hypotheses)} of {self.looked_at}), {self._fit(p)}.{worse} "
                "Shift+C: next, Enter: apply, Esc: cancel."
            )
        times = [p.time for p in self.proposals]
        span = f"t{min(times)}" if min(times) == max(times) else f"t{min(times)}-t{max(times)}"
        parts = [
            f"Cuts for {len(self.proposals)} of {len(self.selection)} selected nodes ({span})"
        ]
        if self.skipped:
            shown = ", ".join(str(n) for n in self.skipped[:5])
            more = f" +{len(self.skipped) - 5}" if len(self.skipped) > 5 else ""
            parts.append(f"none for {shown}{more}")
        worse = [p.node for p in self.proposals if self._worse(p)]
        if worse:
            parts.append(f"{len(worse)} fit their neighbouring frames worse than left whole")
        several = sum(len(p.cuts) > 1 for p in self.proposals)
        step = f"Shift+C: next cut ({several} have others), " if several else ""
        return "; ".join(parts) + f". {step}Enter: apply all, Esc: cancel."

    def announce(self):
        show_info(self.message())

    def draw(self):
        """(Re)draw the current candidates where their nodes are shown now: in the
        kymograph, those on the page shown."""
        if not self.active:
            return
        tracks_viewer = self.tracks_viewer
        viewer = tracks_viewer.viewer
        kymograph = tracks_viewer.kymograph_layers
        in_kymograph = tracks_viewer.view_mode == "kymograph" and kymograph.geometry is not None
        shown = [
            p for p in self.proposals
            if not in_kymograph
            or kymograph.page_start <= p.time < kymograph.page_start + kymograph.page_length
        ]
        if not shown:
            if self.layer is not None and self.layer in viewer.layers:
                viewer.layers.remove(self.layer)
            self.layer = None
            return
        first = min(p.time for p in shown)
        frames = max(p.time for p in shown) - first + 1
        height, width = shown[0].current.pieces[0].shape
        stack = np.zeros((frames, height, width), dtype=np.int32)
        colours = {None: "transparent", 0: "transparent"}
        for n, proposal in enumerate(shown):
            for k, piece in enumerate(proposal.current.pieces):
                value = n * SLOTS + k + 1
                stack[proposal.time - first][piece] = value
                colours[value] = COLOURS[k % len(COLOURS)]
        if in_kymograph:
            # frames side by side along x, like the kymograph page
            geometry = kymograph.geometry
            data = stack.transpose(1, 0, 2).reshape(height, frames * width)
            scale = (geometry.y_scale, geometry.x_scale)
            offset = (first - kymograph.page_start) * geometry.x_size * geometry.x_scale
            translate = (0.0, float(offset))
        else:
            seg = tracks_viewer.tracking_layers.seg_layer
            scale = tuple(seg.scale) if seg is not None else (1.0,) * 3
            base = tuple(seg.translate) if seg is not None else (0.0,) * 3
            data = stack
            translate = (base[0] + first * scale[0], *base[1:])
        colormap = DirectLabelColormap(color_dict=colours)
        if self.layer is not None and self.layer in viewer.layers and self.layer.ndim == data.ndim:
            self.layer.data = data
            self.layer.scale, self.layer.translate = scale, translate
            self.layer.colormap = colormap
            return
        if self.layer is not None and self.layer in viewer.layers:
            viewer.layers.remove(self.layer)
        active = viewer.layers.selection.active
        self.layer = viewer.add_labels(
            data,
            name=LAYER_NAME,
            scale=scale,
            translate=translate,
            opacity=1.0,
            blending="translucent_no_depth",
            colormap=colormap,
        )
        self.layer.contour = 2
        # the user keeps working in the layer they were in
        if active is not None:
            viewer.layers.selection.active = active
