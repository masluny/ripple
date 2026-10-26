#!/usr/bin/env bash
# Install (or reinstall) the Ripple data collectors as per-user launchd agents.
#   scripts/install_collectors.sh            install + start
#   scripts/install_collectors.sh uninstall  stop + remove
# Logs: data/logs/<job>.log. The Mac must be awake for jobs to run; missed StartInterval runs are
# skipped, the daily job runs at the next wake-up after 04:30.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/.venv/bin/ripple"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$REPO/data/logs"
JOBS=(states taf daily)

label() { echo "pl.kotowski.ripple.$1"; }

uninstall() {
  for j in "${JOBS[@]}"; do
    launchctl bootout "gui/$(id -u)/$(label "$j")" 2>/dev/null || true
    rm -f "$AGENTS/$(label "$j").plist"
  done
  echo "collectors removed"
}

if [[ "${1:-}" == "uninstall" ]]; then uninstall; exit 0; fi

[[ -x "$BIN" ]] || { echo "run 'uv sync' first"; exit 1; }
mkdir -p "$AGENTS" "$LOGS"

write_plist() {  # name, args, schedule-xml
  local name=$1 args=$2 schedule=$3
  cat >"$AGENTS/$(label "$name").plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$(label "$name")</string>
  <key>ProgramArguments</key>
  <array><string>$BIN</string>$args</array>
  <key>WorkingDirectory</key><string>$REPO</string>
  $schedule
  <key>StandardOutPath</key><string>$LOGS/$name.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$name.log</string>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
EOF
}

# 270 s -> 320 anonymous credits/day, under OpenSky's 400/day limit with room for manual use.
write_plist states "<string>collect</string><string>states</string>" \
  "<key>StartInterval</key><integer>270</integer><key>RunAtLoad</key><true/>"
write_plist taf "<string>collect</string><string>taf</string>" \
  "<key>StartInterval</key><integer>3600</integer><key>RunAtLoad</key><true/>"
write_plist daily "<string>-v</string><string>collect</string><string>daily</string>" \
  "<key>StartCalendarInterval</key><dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>30</integer></dict>"

for j in "${JOBS[@]}"; do
  launchctl bootout "gui/$(id -u)/$(label "$j")" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$AGENTS/$(label "$j").plist"
done
launchctl list | grep ripple || true
echo "collectors installed; logs in $LOGS"
