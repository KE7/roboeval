#!/usr/bin/env bash
# scripts/release.sh — roboeval release helper
#
# Usage:
#   ./scripts/release.sh <new-version>
#
# What it does:
#   1. Validates a clean working tree.
#   2. Bumps version in pyproject.toml and CITATION.cff.
#   3. Prepends a CHANGELOG.md template entry for the new version.
#   4. Creates a signed git tag v<version>.
#   5. Opens a GitHub release draft (requires `gh` CLI).
#
# Prerequisites: git, gh (GitHub CLI), sed, date.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Argument validation
# ---------------------------------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <new-version>  (e.g. 0.2.0)" >&2
    exit 1
fi

VERSION="$1"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be in MAJOR.MINOR.PATCH format (got '$VERSION')." >&2
    exit 1
fi

TAG="v${VERSION}"
TODAY="$(date -u +%Y-%m-%d)"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# ---------------------------------------------------------------------------
# 2. Require clean working tree
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is dirty. Commit or stash changes before releasing." >&2
    git status --short
    exit 1
fi

if git rev-parse "$TAG" &>/dev/null; then
    echo "Error: tag '$TAG' already exists." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Bump version in pyproject.toml
# ---------------------------------------------------------------------------
echo "Bumping version in pyproject.toml → $VERSION"
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml

# ---------------------------------------------------------------------------
# 4. Bump version in CITATION.cff
# ---------------------------------------------------------------------------
echo "Bumping version in CITATION.cff → $VERSION"
sed -i "s/^version: .*/version: ${VERSION}/" CITATION.cff
sed -i "s/^date-released: .*/date-released: \"${TODAY}\"/" CITATION.cff

# ---------------------------------------------------------------------------
# 5. Prepend CHANGELOG.md template entry
# ---------------------------------------------------------------------------
echo "Prepending CHANGELOG.md entry for [$VERSION] - $TODAY"
CHANGELOG_ENTRY="## [${VERSION}] - ${TODAY}

### Added

- TODO

### Changed

- TODO

### Fixed

- TODO

### Removed

- TODO

"

# Insert after the "## [Unreleased]" line
python3 - <<PYEOF
import re, pathlib

path = pathlib.Path("CHANGELOG.md")
text = path.read_text()

marker = "## [Unreleased]"
insert = """${CHANGELOG_ENTRY}"""

if marker in text:
    text = text.replace(marker, marker + "\n\n" + insert.strip() + "\n", 1)
else:
    # No [Unreleased] section — insert after the first H1 line
    text = re.sub(r"(# .+\n)", r"\1\n" + insert, text, count=1)

path.write_text(text)
PYEOF

# Add comparison link at the bottom of CHANGELOG.md
PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || echo "")"
if [[ -n "$PREV_TAG" ]]; then
    COMPARE_URL="https://github.com/KE7/roboeval/compare/${PREV_TAG}...${TAG}"
    echo "[${VERSION}]: ${COMPARE_URL}" >> CHANGELOG.md
else
    echo "[${VERSION}]: https://github.com/KE7/roboeval/releases/tag/${TAG}" >> CHANGELOG.md
fi

# ---------------------------------------------------------------------------
# 6. Stage the changed files and create a release commit
# ---------------------------------------------------------------------------
git add pyproject.toml CITATION.cff CHANGELOG.md
git commit -m "chore: release ${TAG}"

# ---------------------------------------------------------------------------
# 7. Create git tag
# ---------------------------------------------------------------------------
echo "Creating tag $TAG"
git tag -s "$TAG" -m "Release ${TAG}" 2>/dev/null || git tag "$TAG" -m "Release ${TAG}"

echo ""
echo "Tag '$TAG' created. Push with:"
echo "  git push origin main && git push origin $TAG"
echo ""

# ---------------------------------------------------------------------------
# 8. Open GitHub release draft
# ---------------------------------------------------------------------------
if command -v gh &>/dev/null; then
    echo "Opening GitHub release draft for $TAG …"
    gh release create "$TAG" \
        --title "roboeval ${TAG}" \
        --notes-from-tag \
        --draft \
        --generate-notes
else
    echo "gh CLI not found — skipping release draft creation."
    echo "Create the release manually at: https://github.com/KE7/roboeval/releases/new?tag=${TAG}"
fi
