#!/bin/bash
# Git Commit and Push with Auto-Restage
# Handles nbdev pre-commit hooks that modify files during commit
#
# Usage: ./scripts/git_commit.sh "Your commit message"

set -e

MESSAGE="$1"

if [ -z "$MESSAGE" ]; then
    echo "Usage: ./scripts/git_commit.sh \"Your commit message\""
    exit 1
fi

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}[1/5] Staging all changes...${NC}"
git add -A

echo -e "${CYAN}[2/5] Regenerating LLM context file...${NC}"
if [ -f "scripts/local_ctx.py" ]; then
    python scripts/local_ctx.py
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ llms-ctx.txt updated${NC}"
        git add llms-ctx.txt 2>/dev/null || true
    fi
fi

echo -e "${CYAN}[3/5] Running nbdev_prepare...${NC}"
nbdev_prepare
git add -A

echo -e "${CYAN}[4/5] Committing (pre-commit hooks will run)...${NC}"
if ! git commit -m "$MESSAGE"; then
    echo -e "${YELLOW}[4b/5] Pre-commit hooks modified files, re-staging and committing...${NC}"
    git add -A
    git commit -m "$MESSAGE" || {
        echo -e "${RED}✗ Commit failed after restage${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Commit successful${NC}"

echo -e "${CYAN}[5/5] Pushing to remote...${NC}"
if ! git push 2>/dev/null; then
    echo -e "${YELLOW}    Setting upstream branch...${NC}"
    git push -u origin main
fi

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All done! Check GitHub Actions: https://github.com/abhisheksreesaila/mxframe/actions${NC}"
else
    echo -e "\n${RED}✗ Push failed!${NC}"
    exit 1
fi
