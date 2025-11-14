#!/bin/bash
# Rollback script for Phase 1 (Ruff Migration)
# This script reverts changes from Phase 1 if issues are discovered

set -e  # Exit on error

echo "🔄 Rolling back Phase 1 (Ruff Migration)..."
echo ""

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Confirm rollback
read -p "⚠️  This will revert all Ruff changes. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Rollback cancelled"
    exit 1
fi

# Step 1: Restore Black configuration in pyproject.toml
echo "1. Restoring Black configuration in pyproject.toml..."
if grep -q "\[tool.ruff\]" pyproject.toml; then
    echo "   Removing Ruff configuration..."
    # This will need manual editing - just warn user
    echo "   ⚠️  WARNING: Please manually restore Black config in pyproject.toml"
    echo "   Ruff config detected - manual cleanup required"
else
    echo "   ✅ No Ruff config found"
fi

# Step 2: Reformat all code with Black
echo "2. Reformatting code with Black..."
if command -v black &> /dev/null; then
    black phentrieve/ api/ tests/
    echo "   ✅ Code reformatted with Black"
else
    echo "   ❌ Error: Black not installed"
    exit 1
fi

# Step 3: Reinstall Black if it was removed
echo "3. Ensuring Black is installed..."
pip install black
echo "   ✅ Black installed"

# Step 4: Remove Ruff if it was added to dependencies
echo "4. Checking Ruff installation..."
if pip show ruff &> /dev/null; then
    echo "   Ruff is installed (keeping it for now - can uninstall manually if needed)"
else
    echo "   ✅ Ruff not in pip packages"
fi

# Step 5: Restore any reverted linting fixes
echo "5. Reverting linting changes..."
echo "   ⚠️  WARNING: Auto-fixed code changes cannot be automatically reverted"
echo "   You may need to:"
echo "   - git revert the Phase 1 commits, or"
echo "   - git checkout the pre-Phase-1 commit"
echo ""

# Step 6: Update Makefile if needed
if [ -f "Makefile" ]; then
    echo "6. Checking Makefile..."
    if grep -q "ruff" Makefile; then
        echo "   ⚠️  WARNING: Makefile contains Ruff commands"
        echo "   Please manually update Makefile to use Black instead"
    else
        echo "   ✅ Makefile OK"
    fi
fi

echo ""
echo "✅ Phase 1 rollback steps completed!"
echo ""
echo "⚠️  MANUAL STEPS REQUIRED:"
echo "   1. Restore Black configuration in pyproject.toml:"
echo "      [tool.black]"
echo "      line-length = 88"
echo "      target-version = ['py39']"
echo "      include = '\\.pyi?$'"
echo ""
echo "   2. If code quality issues remain, revert commits:"
echo "      git revert <phase-1-commit-hash>"
echo ""
echo "   3. Update Makefile to use Black instead of Ruff"
echo ""
echo "   4. Run tests to verify rollback:"
echo "      pytest tests/"
echo ""
echo "   5. Verify formatting:"
echo "      black phentrieve/ api/ tests/ --check"
echo ""
