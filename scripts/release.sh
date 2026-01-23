#!/bin/bash
# ========================================
# MXFRAME RELEASE & PUBLISH SCRIPT
# ========================================
# Prerequisites:
#   - GitHub token: export GITHUB_TOKEN=ghp_...
#   - PyPI token in ~/.pypirc or pass to twine
#
# PRE-RELEASE CHECKLIST:
#   1. nbdev_test
#   2. nbdev_prepare  
#   3. nbdev_preview (verify docs)
#   4. ./scripts/git_commit.sh "Release v0.0.x"
#   5. Wait for GitHub Actions to pass
#
# Usage: ./scripts/release.sh
# ========================================

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "\n${CYAN}===== MXFRAME RELEASE & PUBLISH =====${NC}"

# Step 1: Check GitHub token
echo -e "\n${YELLOW}[1/4] Checking GitHub token...${NC}"
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}  ✗ GITHUB_TOKEN not set!${NC}"
    echo -e "${YELLOW}  Set it with: export GITHUB_TOKEN=ghp_...${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ GitHub token found${NC}"

# Step 2: Create GitHub release
echo -e "\n${YELLOW}[2/4] Creating GitHub release...${NC}"
nbdev_release_git
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ GitHub release created${NC}"
    echo -e "  View at: https://github.com/abhisheksreesaila/mxframe/releases"
else
    echo -e "${RED}  ✗ GitHub release failed${NC}"
    echo -e "${YELLOW}  Check that version in settings.ini was updated${NC}"
    exit 1
fi

# Step 3: Build Python package
echo -e "\n${YELLOW}[3/4] Building Python package...${NC}"
rm -rf dist/ 2>/dev/null || true
python -m build
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Package built in dist/${NC}"
    ls -la dist/
else
    echo -e "${RED}  ✗ Build failed${NC}"
    echo -e "${YELLOW}  Install build: pip install build${NC}"
    exit 1
fi

# Step 4: Upload to PyPI
echo -e "\n${YELLOW}[4/4] Uploading to PyPI...${NC}"
echo -e "  Publishing mxframe to https://pypi.org/project/mxframe/"
twine upload dist/*
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Successfully released to PyPI!${NC}"
else
    echo -e "${RED}  ✗ PyPI upload failed${NC}"
    echo -e "${YELLOW}  Install twine: pip install twine${NC}"
    echo -e "${YELLOW}  Or use: twine upload dist/* --username __token__ --password pypi-...${NC}"
    exit 1
fi

echo -e "\n${GREEN}===== RELEASE COMPLETE =====${NC}"
echo -e "✓ GitHub: https://github.com/abhisheksreesaila/mxframe/releases"
echo -e "✓ PyPI: https://pypi.org/project/mxframe/"
echo -e "✓ Docs: https://abhisheksreesaila.github.io/mxframe/"
echo -e "\nInstall with: ${CYAN}pip install mxframe${NC}"
