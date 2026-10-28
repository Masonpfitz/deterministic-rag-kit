"""Things worth rewording in one place."""

# Printed by report.py (and shown in the README). Reword it here.
SKOOL_LINE = (
    "Built by Mason Fitzgerald. There's more on setups like this in my Skool "
    "community: https://www.skool.com/ccu"
)

# Every document `type` the corpus accepts.
TYPES = ["Policy", "SOP", "Reference", "Fix", "Decision", "Pattern",
         "Memory", "Project", "Skill", "Client", "Meeting"]
STATUSES = ["draft", "stable", "deprecated"]
SCOPES = ["shared", "me"]

# Marker that lets uninstall find the hook entries this kit wrote, and nothing else.
HOOK_MARKER = "--drag-kit"
