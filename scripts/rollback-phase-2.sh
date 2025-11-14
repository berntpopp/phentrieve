#!/bin/bash
# Rollback script for Phase 2 (uv Migration)
# This script reverts changes from Phase 2 if issues are discovered

set -e  # Exit on error

echo "🔄 Rolling back Phase 2 (uv Migration)..."
echo ""

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Confirm rollback
read -p "⚠️  This will revert all uv changes. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Rollback cancelled"
    exit 1
fi

# Step 1: Check if baseline requirements exist
echo "1. Checking for baseline requirements..."
if [ ! -f "plan/requirements-baseline.txt" ]; then
    echo "   ❌ Error: plan/requirements-baseline.txt not found"
    echo "   Cannot rollback without baseline requirements file"
    exit 1
fi
echo "   ✅ Baseline requirements found"

# Step 2: Deactivate current environment
echo "2. Deactivating current environment..."
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
    echo "   ✅ Environment deactivated"
else
    echo "   ℹ️  No virtual environment active"
fi

# Step 3: Remove uv-created virtual environment
echo "3. Removing uv virtual environment..."
if [ -d ".venv" ]; then
    echo "   Removing .venv directory..."
    rm -rf .venv
    echo "   ✅ .venv removed"
else
    echo "   ℹ️  No .venv directory found"
fi

# Step 4: Remove uv lockfile
echo "4. Removing uv lockfile..."
if [ -f "uv.lock" ]; then
    rm uv.lock
    echo "   ✅ uv.lock removed"
else
    echo "   ℹ️  No uv.lock found"
fi

# Step 5: Restore pip-based virtual environment
echo "5. Creating pip-based virtual environment..."
python -m venv .venv
echo "   ✅ Virtual environment created"

# Step 6: Activate new environment
echo "6. Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate
echo "   ✅ Environment activated"

# Step 7: Install from baseline requirements
echo "7. Installing dependencies from baseline..."
echo "   This may take several minutes..."
pip install --upgrade pip
pip install -r plan/requirements-baseline.txt
echo "   ✅ Dependencies installed from baseline"

# Step 8: Install package in development mode
echo "8. Installing package in development mode..."
pip install -e .
echo "   ✅ Package installed"

# Step 9: Restore pyproject.toml if needed
echo "9. Checking pyproject.toml..."
if grep -q "\[tool.uv\]" pyproject.toml; then
    echo "   ⚠️  WARNING: pyproject.toml contains uv configuration"
    echo "   Please manually remove [tool.uv] section and uv-specific settings"
else
    echo "   ✅ No uv config found in pyproject.toml"
fi

# Step 10: Update Makefile if needed
if [ -f "Makefile" ]; then
    echo "10. Checking Makefile..."
    if grep -q "uv sync" Makefile; then
        echo "    ⚠️  WARNING: Makefile contains uv commands"
        echo "    Please manually update Makefile to use pip instead"
    else
        echo "    ✅ Makefile OK"
    fi
fi

# Step 11: Verify installation
echo "11. Verifying installation..."
if python -c "import phentrieve" 2>/dev/null; then
    echo "    ✅ Package imports successfully"
else
    echo "    ❌ Error: Package import failed"
    exit 1
fi

echo ""
echo "✅ Phase 2 rollback completed!"
echo ""
echo "📋 VERIFICATION STEPS:"
echo "   1. Run tests:"
echo "      pytest tests/"
echo ""
echo "   2. Verify CLI works:"
echo "      phentrieve --help"
echo ""
echo "   3. Check installed packages:"
echo "      pip list > current-packages.txt"
echo "      diff plan/requirements-baseline.txt current-packages.txt"
echo ""
echo "   4. If needed, manually restore pyproject.toml:"
echo "      - Remove [tool.uv] section"
echo "      - Remove uv-specific project.dependencies format"
echo "      - Restore original dependencies format"
echo ""
echo "⚠️  MANUAL STEPS REQUIRED:"
echo "   1. Update Makefile to use pip instead of uv"
echo "   2. Update CI/CD workflows to use pip instead of uv"
echo "   3. Remove uv configuration from pyproject.toml"
echo "   4. Document version differences if any issues occurred"
echo ""
echo "💡 TIP: Keep uv.lock from Phase 2 for reference if issues were version-related"
echo ""
