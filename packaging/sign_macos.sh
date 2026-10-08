#!/usr/bin/env bash
#
# Sign the macOS app bundle, inside-out.
#
# Usage:  packaging/sign_macos.sh "dist/Squeak Peek Studio.app"
#
# Identity comes from $MACOS_SIGN_IDENTITY. If that is unset or empty the
# script falls back to an ad-hoc signature ("-"), which is what an unsigned
# release build gets: it satisfies the macOS 11+ requirement that every
# binary carry *some* signature, but it is not a Developer ID signature and
# Gatekeeper will still refuse it on another machine without the
# right-click-Open workaround. See packaging/SIGNING.md.
#
# Why not `codesign --deep`: Apple documents --deep as a verification-only
# convenience and explicitly advises against signing with it (it applies the
# top-level entitlements to nested code and skips bundles it does not
# recognise). The supported approach, implemented below, is to sign every
# nested Mach-O first, then nested framework bundles deepest-first, then the
# outer bundle last.
#
# entitlements.plist must stay comment-free: the kernel's AMFI plist parser
# rejects XML comments outright. The rationale for each entitlement, and
# which ones are verified load-bearing, is in packaging/SIGNING.md.
set -euo pipefail

APP="${1:?usage: sign_macos.sh <path to .app>}"
[ -d "$APP" ] || { echo "error: no such bundle: $APP" >&2; exit 1; }

PACKAGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTITLEMENTS="$PACKAGING_DIR/entitlements.plist"

IDENTITY="${MACOS_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
    IDENTITY="-"
    echo "==> No MACOS_SIGN_IDENTITY set; signing ad-hoc (not distributable)."
else
    echo "==> Signing with identity: $IDENTITY"
fi

# Strip quarantine/resource-fork xattrs; codesign fails on extended
# attributes that survive the artifact download/unzip round-trip.
xattr -cr "$APP"

FLAGS=(--force --options runtime --entitlements "$ENTITLEMENTS")
if [ "$IDENTITY" = "-" ]; then
    # Ad-hoc signatures cannot carry a trusted timestamp; asking for one
    # makes codesign emit a warning per binary and changes nothing.
    FLAGS+=(--timestamp=none)
else
    FLAGS+=(--timestamp)
fi

sign_one() {
    codesign "${FLAGS[@]}" --sign "$IDENTITY" "$1"
}

# 1. Every loose Mach-O inside the bundle (dylibs, .so extension modules,
#    helper executables), excluding anything inside a .framework — those are
#    signed as whole bundles in step 2.
echo "==> Signing nested Mach-O files"
count=0
while IFS= read -r -d '' f; do
    case "$f" in
        *.framework/*) continue ;;
    esac
    if file -b "$f" | grep -q 'Mach-O'; then
        sign_one "$f"
        count=$((count + 1))
    fi
done < <(find "$APP/Contents" -type f -print0)
echo "    signed $count nested binaries"

# 2. Nested framework bundles, deepest first (-depth), so an inner framework
#    is sealed before the framework that contains it.
echo "==> Signing nested frameworks"
while IFS= read -r -d '' fw; do
    sign_one "$fw"
done < <(find "$APP" -name '*.framework' -type d -depth -print0)

# 3. The app bundle itself, last.
echo "==> Signing app bundle"
sign_one "$APP"

echo "==> Verifying"
# --deep IS the right flag for verification (unlike signing); quiet on success.
codesign --verify --deep --strict "$APP"
codesign --display --entitlements - "$APP" >/dev/null

if [ "$IDENTITY" = "-" ]; then
    echo "==> Ad-hoc signature valid (Gatekeeper will still block on other Macs)."
else
    # Only meaningful once the bundle is also notarized+stapled; a plain
    # Developer ID signature fails this until then, so it is informational.
    spctl --assess --type exec --verbose=4 "$APP" || \
        echo "    (spctl rejects until the DMG is notarized and stapled — expected here)"
fi
