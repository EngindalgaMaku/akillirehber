#!/bin/bash
# Test script for auto-restore functionality
# This script helps verify that auto-restore will work on deployment

set -e

echo "=========================================="
echo "Auto-Restore Test Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backups directory exists
echo "1. Checking backups directory..."
if [ -d "./backups" ]; then
    echo -e "${GREEN}✓${NC} backups/ directory exists"
else
    echo -e "${RED}✗${NC} backups/ directory not found"
    exit 1
fi

# Check for PostgreSQL backup
echo ""
echo "2. Checking for PostgreSQL backup..."
PG_BACKUP=$(ls -t ./backups/postgres-*.sql 2>/dev/null | head -1)
if [ -n "$PG_BACKUP" ]; then
    echo -e "${GREEN}✓${NC} Found: $(basename $PG_BACKUP)"
    echo "   Size: $(du -h $PG_BACKUP | cut -f1)"
    echo "   Modified: $(stat -c %y $PG_BACKUP 2>/dev/null || stat -f %Sm $PG_BACKUP 2>/dev/null)"
else
    echo -e "${YELLOW}⚠${NC} No PostgreSQL backup found (postgres-*.sql)"
fi

# Check for Weaviate backup
echo ""
echo "3. Checking for Weaviate backup..."
WV_BACKUP=$(ls -t ./backups/weaviate-*.json 2>/dev/null | head -1)
if [ -n "$WV_BACKUP" ]; then
    echo -e "${GREEN}✓${NC} Found: $(basename $WV_BACKUP)"
    echo "   Size: $(du -h $WV_BACKUP | cut -f1)"
    echo "   Modified: $(stat -c %y $WV_BACKUP 2>/dev/null || stat -f %Sm $WV_BACKUP 2>/dev/null)"
    
    # Check JSON validity
    if python3 -c "import json; json.load(open('$WV_BACKUP'))" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} JSON is valid"
        
        # Show collection count
        COLLECTIONS=$(python3 -c "import json; data=json.load(open('$WV_BACKUP')); print(len(data.get('collections', {})))")
        echo "   Collections: $COLLECTIONS"
    else
        echo -e "${RED}✗${NC} JSON is invalid"
    fi
else
    echo -e "${YELLOW}⚠${NC} No Weaviate backup found (weaviate-*.json)"
fi

# Check docker-compose.yml volume mount
echo ""
echo "4. Checking docker-compose.yml volume mount..."
if grep -q "./backups:/app/backups" docker-compose.yml; then
    echo -e "${GREEN}✓${NC} Volume mount found in docker-compose.yml"
else
    echo -e "${RED}✗${NC} Volume mount missing in docker-compose.yml"
    echo "   Add this to backend service volumes:"
    echo "   - ./backups:/app/backups:ro"
fi

# Check docker-compose.coolify.yml volume mount
echo ""
echo "5. Checking docker-compose.coolify.yml volume mount..."
if grep -q "./backups:/app/backups" docker-compose.coolify.yml; then
    echo -e "${GREEN}✓${NC} Volume mount found in docker-compose.coolify.yml"
else
    echo -e "${RED}✗${NC} Volume mount missing in docker-compose.coolify.yml"
    echo "   Add this to backend service volumes:"
    echo "   - ./backups:/app/backups:ro"
fi

# Check entrypoint script
echo ""
echo "6. Checking entrypoint script..."
if [ -f "backend/docker-entrypoint.sh" ]; then
    echo -e "${GREEN}✓${NC} backend/docker-entrypoint.sh exists"
    
    if grep -q "Auto-restore from backups" backend/docker-entrypoint.sh; then
        echo -e "${GREEN}✓${NC} Auto-restore logic found"
    else
        echo -e "${RED}✗${NC} Auto-restore logic not found"
    fi
else
    echo -e "${RED}✗${NC} backend/docker-entrypoint.sh not found"
fi

# Check .gitignore
echo ""
echo "7. Checking .gitignore..."
if grep -q "^backups/" .gitignore; then
    echo -e "${YELLOW}⚠${NC} backups/ is in .gitignore (backups won't be committed)"
    echo "   Comment out 'backups/' in .gitignore to commit backups"
elif grep -q "# backups/" .gitignore; then
    echo -e "${GREEN}✓${NC} backups/ is commented in .gitignore (backups will be committed)"
else
    echo -e "${GREEN}✓${NC} backups/ not in .gitignore (backups will be committed)"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

READY=true

if [ -z "$PG_BACKUP" ]; then
    echo -e "${YELLOW}⚠${NC} No PostgreSQL backup - create one at http://localhost:3000/dashboard/backup"
    READY=false
fi

if [ -z "$WV_BACKUP" ]; then
    echo -e "${YELLOW}⚠${NC} No Weaviate backup - create one at http://localhost:3000/dashboard/backup"
    READY=false
fi

if ! grep -q "./backups:/app/backups" docker-compose.coolify.yml; then
    echo -e "${RED}✗${NC} Volume mount missing in docker-compose.coolify.yml"
    READY=false
fi

if $READY; then
    echo ""
    echo -e "${GREEN}✓ Auto-restore is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. git add backups/"
    echo "2. git commit -m 'Add database backups for auto-restore'"
    echo "3. git push origin main"
    echo "4. Deploy to Coolify"
    echo ""
    echo "On first deployment, the system will automatically:"
    echo "- Check if database is empty"
    echo "- Restore PostgreSQL from: $(basename $PG_BACKUP)"
    echo "- Restore Weaviate from: $(basename $WV_BACKUP)"
    echo "- Start application with restored data"
else
    echo ""
    echo -e "${YELLOW}⚠ Auto-restore is not ready${NC}"
    echo ""
    echo "Fix the issues above and run this script again."
fi

echo ""
echo "=========================================="
