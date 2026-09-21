# Motile Tracker: install and annotate (macOS)

## Install (once, about 5 minutes, needs internet)

1. Download **Install Motile Tracker.command** and put it in your Downloads folder.
2. Right-click it, choose **Open**, then **Open** again in the security dialog.
   (macOS blocks double-clicking downloaded scripts. If the dialog only offers
   "Move to Bin", open **System Settings > Privacy & Security**, scroll down and
   click **Open Anyway**.)
3. A Terminal window shows progress. Wait for **Done**, then close it.
4. **Motile Tracker** is now in your Applications folder (Finder > Go >
   Applications, or search Spotlight). The first start takes about 30 seconds.

Running the installer again updates to the latest version.

## Load your data

1. Copy the `.zarr` folder you were given somewhere on your laptop (for example
   Documents). Do not rename anything inside it.
2. In Motile Tracker, open the **Kymograph Data** tab on the right, click
   **Browse...** and choose the `.zarr` folder.
3. Pick a trench from the dropdown and click **Load**. The kymograph appears:
   every frame side by side, cells coloured by track, lines linking a cell to
   itself in the next frame.

## Look around

- **Time slider** under the image: drag it, or use its arrow buttons, to move one
  frame at a time. The view turns the page when needed.
- **Scroll** to zoom, **drag** to pan. Home button (bottom left) resets the view.
- **Q** cycles the display: all tracks / only the selected lineage.
- **Visualization** tab: how many frames per page (smaller pages are easier to
  read), and which link types to show.

## Fix errors (one key each)

Select a cell by clicking its coloured mask. Shift-click adds a second cell.

| Problem | Fix |
|---|---|
| One mask covers two cells | Select it, press **C** (Split). Both halves are selected so you can check the cut. |
| Two masks for one cell | Click one, shift-click the other, press **J** (Merge). |
| Wrong link between two cells | Select both, press **B** (Break). |
| Missing link between two cells | Select both (any frames), press **A** (Add). A dashed line shows a link that skips frames. |
| A cell that is not a cell | Select it, press **D** (Delete). |
| Two cells swapped identity | Select both, press **S** (Swap). |
| Anything you regret | **Z** undoes one step; **R** redoes. |

Messages in the bottom right corner tell you when a link could not be made
(for example a cell that already has two daughters).

## Save your work

Nothing is saved automatically. In the **Tracks List** tab, click the **save**
icon on the row of your trench. It writes a folder ending in `.geff` into the
directory shown above the list (Browse to change it, for example to a shared
drive). Save every 10 to 15 minutes, and once more before quitting.

To check a save worked: the folder's modification time in Finder updates, and
loading it back (below) shows your edits.

To continue later: Tracks List > choose **Tracks (geff)** in the dropdown >
**Load**, pick the saved `.geff` folder, then load the same trench's image in the
Kymograph Data tab with *Open in kymograph view* unticked, select the image layer
in the Visualization tab, and switch to the kymograph.
