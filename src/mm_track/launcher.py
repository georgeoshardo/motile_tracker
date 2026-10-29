import napari

# Auto-load your plugin
viewer = napari.Viewer()
viewer.window.add_plugin_dock_widget("mm-track")

# Start Napari event loop
napari.run()
