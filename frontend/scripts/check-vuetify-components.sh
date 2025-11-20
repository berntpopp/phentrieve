#!/bin/bash
# Script to find all Vuetify components used in the application
# Helps prevent regressions when tree-shaking

echo "=== Vuetify Components Used in Application ==="
echo ""

# Find all v- tags in Vue files
echo "Components found in templates:"
grep -roh "<v-[a-z-]*" src/ --include="*.vue" | sed 's/<//g' | sort -u

echo ""
echo "=== Directives used (v-tooltip, v-ripple) ==="
grep -roh "v-tooltip\|v-ripple\|v-intersect\|v-resize\|v-scroll\|v-touch" src/ --include="*.vue" | sort -u

echo ""
echo "Done! Compare this list with src/plugins/vuetify.js"
