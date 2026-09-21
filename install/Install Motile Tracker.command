#!/bin/bash
# Installs Motile Tracker (kymograph fork) on macOS without developer tools.
# Double-click this file. It installs the uv package manager for the current
# user, downloads the tracker, builds its Python environment and puts a
# "Motile Tracker" app into ~/Applications. Safe to run again to update.
set -euo pipefail

REPO_ZIP="${MOTILE_TRACKER_ZIP:-https://github.com/georgeoshardo/motile_tracker/archive/refs/heads/main.zip}"
INSTALL_DIR="${MOTILE_TRACKER_HOME:-$HOME/Library/Application Support/MotileTrackerApp}"
APP_DIR="${MOTILE_TRACKER_APP_DIR:-$HOME/Applications}"
APP="$APP_DIR/Motile Tracker.app"

say_step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

say_step "Installing the uv package manager (this also provides Python)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

say_step "Downloading Motile Tracker"
mkdir -p "$INSTALL_DIR"
TMP="$(mktemp -d)"
curl -L --fail --progress-bar "$REPO_ZIP" -o "$TMP/src.zip"
unzip -q "$TMP/src.zip" -d "$TMP/unpacked"
SRC="$(find "$TMP/unpacked" -mindepth 1 -maxdepth 1 -type d | head -1)"
# keep the environment between updates, replace the source
if [ -d "$INSTALL_DIR/src" ]; then rm -rf "$INSTALL_DIR/src"; fi
mkdir -p "$INSTALL_DIR/src"
cp -R "$SRC"/. "$INSTALL_DIR/src/"
rm -rf "$TMP"

say_step "Building the Python environment (a few minutes the first time)"
cd "$INSTALL_DIR/src"
# the zip carries no git history, from which the package normally derives its version
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MOTILE_TRACKER="5.0.1+kymograph"
uv sync --python 3.13 --no-dev --extra all

say_step "Checking the installation"
uv run --no-sync python -c "import motile_tracker, napari; print('napari', napari.__version__, 'OK')"

say_step "Creating the app in $APP_DIR"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Motile Tracker</string>
  <key>CFBundleDisplayName</key><string>Motile Tracker</string>
  <key>CFBundleIdentifier</key><string>org.motile-tracker.kymograph</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>launch</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/launch" <<LAUNCH
#!/bin/bash
export PATH="\$HOME/.local/bin:\$PATH"
cd "$INSTALL_DIR/src"
exec uv run --no-sync motile_tracker --mode editing "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/launch"
touch "$APP"

say_step "Done"
echo "Motile Tracker is installed. Open it from $APP_DIR (Finder > Go > Applications, or Spotlight: 'Motile Tracker')."
echo "The first launch takes about half a minute while napari starts."
echo "You can close this window."
