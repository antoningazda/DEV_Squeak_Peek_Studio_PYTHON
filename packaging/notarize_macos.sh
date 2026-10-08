#!/usr/bin/env bash
#
# Sign and notarize the release DMG, then staple the ticket to it.
#
# Usage:  packaging/notarize_macos.sh SqueakPeekStudio-macOS.dmg
#
# Requires a Developer ID Application certificate in the keychain plus an
# Apple ID with an app-specific password:
#   MACOS_SIGN_IDENTITY     e.g. "Developer ID Application: Name (TEAMID)"
#   MACOS_NOTARY_APPLE_ID   the Apple ID the membership belongs to
#   MACOS_NOTARY_PASSWORD   app-specific password (appleid.apple.com), NOT
#                           the account password
#   MACOS_NOTARY_TEAM_ID    10-char team identifier
#
# If MACOS_SIGN_IDENTITY is empty this exits 0 without doing anything, so
# the release workflow can call it unconditionally on unsigned builds.
#
# Stapling matters: it embeds the notarization ticket in the DMG so the
# first launch works on a machine with no network access to Apple's
# notarization service.
set -euo pipefail

DMG="${1:?usage: notarize_macos.sh <path to .dmg>}"
[ -f "$DMG" ] || { echo "error: no such file: $DMG" >&2; exit 1; }

if [ -z "${MACOS_SIGN_IDENTITY:-}" ]; then
    echo "==> No signing identity; skipping notarization (unsigned release)."
    exit 0
fi

: "${MACOS_NOTARY_APPLE_ID:?MACOS_NOTARY_APPLE_ID is required to notarize}"
: "${MACOS_NOTARY_PASSWORD:?MACOS_NOTARY_PASSWORD is required to notarize}"
: "${MACOS_NOTARY_TEAM_ID:?MACOS_NOTARY_TEAM_ID is required to notarize}"

echo "==> Signing the disk image"
codesign --force --timestamp --sign "$MACOS_SIGN_IDENTITY" "$DMG"

echo "==> Submitting to Apple notary service (this can take several minutes)"
# --wait blocks until Apple returns Accepted/Invalid; without it the staple
# below would race the submission.
xcrun notarytool submit "$DMG" \
    --apple-id "$MACOS_NOTARY_APPLE_ID" \
    --password "$MACOS_NOTARY_PASSWORD" \
    --team-id "$MACOS_NOTARY_TEAM_ID" \
    --wait

echo "==> Stapling ticket"
xcrun stapler staple "$DMG"

echo "==> Verifying"
xcrun stapler validate "$DMG"
spctl --assess --type open --context context:primary-signature --verbose=4 "$DMG"

echo "==> Notarized and stapled: $DMG"
