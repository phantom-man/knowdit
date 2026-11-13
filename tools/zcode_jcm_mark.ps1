# zcode_jcm_mark.ps1 - ZCode PostToolUse hook. Writes the ONE marker that releases the
# Read/Bash-search guard in zcode_jcm_guard.ps1.
#
# v2 (owner order 2026-10-20, after the W3-session grep drift): ONLY Skill(jcodemunch)
# releases the guard. Bare jCodemunch MCP calls NO LONGER write the marker - the exact
# drift mode was "early resolve_repo calls wrote the marker, every later grep sailed".
# The skill is the entry point because its contract is the health flow the owner
# specified: analyze target indexes -> fix stale ones -> verify the fix -> proceed.
# Skill names may arrive plugin-namespaced (tools:jcodemunch) - match on the last
# colon segment.
#
# Wired in the NEXUSMILL repo's .zcode/config.json (PostToolUse, matcher
# "Skill|mcp__(plugin_jcodemunch_jcodemunch|jcodemunch)__.*" - landed with the G6 v2
# tranche there; THIS Tools copy is the canonical twin SOURCE, not the wired instance):
# Skill events carry the release decision; the MCP matcher row remains so the hook can
# deliberately IGNORE those events (documented non-release). The session id arrives as
# -SessionId (expanded from ${CLAUDE_SESSION_ID} by the hook runner) because ZCode
# payloads carry none.
#
# Fail-open: any error exits 0 and writes nothing (the guard simply keeps blocking, which
# is the safe direction for a rule-enforcement marker).

param([string]$SessionId = "")
$ErrorActionPreference = "Stop"
try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { exit 0 }
    $payload = $raw | ConvertFrom-Json

    $tool = [string]$payload.tool_name

    if ($tool -eq "Skill") {
        $skill = [string]$payload.tool_input.skill
        if (-not $skill) { exit 0 }
        $tail = ($skill -split ':')[-1]
        if ($tail -ne "jcodemunch") { exit 0 }
        # A FAILED skill invocation must NOT release the guard (gate_20261107-023229
        # LOW): the marker used to be written for any Skill event whose name tail
        # matched, so a skill that errored still unlocked Read/Bash-search for the
        # rest of the session - the "health flow completed" precondition the guard
        # documents was never actually checked.
        #
        # BOTH CHECKS, NOT EITHER (gate_20261107-044923). Positive-signal-only
        # was a regression: it dropped the explicit error-field checks, so a
        # response carrying the skill body AND is_error:true would have
        # released the guard. The release now requires the positive signal AND
        # the absence of an explicit error flag. Everything else - absent,
        # error-flagged, unrecognised - writes nothing, which keeps the guard
        # BLOCKING (the safe direction).
        $resp = $payload.tool_response
        if ($null -eq $resp) { exit 0 }
        # 1. explicit error flags win outright
        if ($resp -isnot [string]) {
            $isErr = $false
            if ($resp.PSObject.Properties.Name -contains 'is_error') { $isErr = [bool]$resp.is_error }
            if (-not $isErr -and $resp.PSObject.Properties.Name -contains 'isError') { $isErr = [bool]$resp.isError }
            if (-not $isErr -and $resp.PSObject.Properties.Name -contains 'error') {
                $isErr = [bool]$resp.error
            }
            if ($isErr) { exit 0 }
        }
        # 2. and the response must carry the skill's own body text
        $text = if ($resp -is [string]) { $resp } else { ($resp | ConvertTo-Json -Depth 6 -Compress) }
        if (-not $text) { exit 0 }
        # ConvertTo-Json -Compress escapes newlines as the two literal characters
        # \n, so the multiline frontmatter match below could NEVER fire on an
        # object-shaped response - a successful skill would never release the
        # guard (gate_20261107-044205). Unescape before matching.
        $text = $text -replace '\\r\\n', "`n" -replace '\\n', "`n" -replace '\\t', "`t"
        # The skill body always opens with its frontmatter name line. TWO forms
        # count as the positive signal (gate_20261107-061040): the YAML
        # frontmatter line (`name: jcodemunch`) and the compact-JSON key
        # (`"name":"jcodemunch"`) that ConvertTo-Json produces for an
        # object-shaped response - the unescape above cannot remove the JSON
        # quoting, so an unquoted-only pattern left the object path unreachable.
        if ($text -notmatch '(?m)^\s*name:\s*jcodemunch\s*$' -and
            $text -notmatch '"name"\s*:\s*"jcodemunch"') { exit 0 }
    }
    else {
        # v2: bare jCodemunch MCP calls are deliberately NOT a release - the matcher
        # still routes them here (both mount spellings) so this branch is the
        # documented ignore. A future re-widening is a one-line change at this exit.
        exit 0
    }

    $sid = if ($SessionId) { $SessionId } else { [string]$payload.session_id }
    # the SAME fallback the guard uses: an unexpanded literal ${CLAUDE_SESSION_ID} must resolve
    # to the shared "nosession" marker on BOTH sides or the release path is dead (gate r4 f1)
    if (-not $sid -or $sid -eq '${CLAUDE_SESSION_ID}') { $sid = "nosession" }
    $marker = Join-Path $env:TEMP ("jcmz_used_" + ($sid -replace '[^A-Za-z0-9_\-]', '_') + ".flag")
    Set-Content -LiteralPath $marker -Value (Get-Date -Format s) -ErrorAction SilentlyContinue
    exit 0
}
catch {
    exit 0
}
