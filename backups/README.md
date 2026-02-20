# Backups Directory

This directory contains database and vector store backups that can be automatically restored on deployment.

## Auto-Restore Feature

When you deploy the application to a fresh environment (e.g., Coolify), the system will automatically check this directory for backups and restore them if the database is empty.

### How It Works

1. **On First Deployment**: The `backend/docker-entrypoint.sh` script checks if the database is empty
2. **If Empty**: It looks for backup files in `/app/backups` (mapped from `./backups`)
3. **Auto-Restore**: Automatically restores the latest PostgreSQL and Weaviate backups
4. **Skip on Subsequent Runs**: If database has data, auto-restore is skipped

### Backup File Naming

The auto-restore script looks for files with these patterns:

- **PostgreSQL**: `postgres-*.sql` (e.g., `postgres-20260205-143000.sql`)
- **Weaviate**: `weaviate-*.json` (e.g., `weaviate-20260205-143000.json`)

It automatically selects the **latest** backup based on file modification time.

## Creating Backups

### From Web Panel

1. Login as admin user
2. Go to `/dashboard/backup`
3. Click "Create PostgreSQL Backup" or "Create Weaviate Backup"
4. Download the backup files
5. Copy them to this `./backups` directory
6. Commit to GitHub

### From Command Line

You can also use the sync scripts to transfer backups from localhost to Coolify:

```bash
# Windows
.\sync-to-coolify.ps1 -SyncBackups

# Linux/Mac
./sync-to-coolify.sh --backups
```

## Deployment Workflow

### Step 1: Create Backups on Localhost

```bash
# Access the backup page
http://localhost:3000/dashboard/backup

# Create backups and download them
# Files will be saved to /var/backups/rag/ in the container
```

### Step 2: Copy Backups to Project

```bash
# Copy downloaded backups to this directory
cp ~/Downloads/postgres-*.sql ./backups/
cp ~/Downloads/weaviate-*.json ./backups/
```

### Step 3: Commit to GitHub

```bash
git add backups/
git commit -m "Add database backups for deployment"
git push origin main
```

### Step 4: Deploy to Coolify

When Coolify pulls the latest code and starts the containers:

1. Backend container starts
2. Entrypoint script runs
3. Checks if database is empty
4. Finds backups in `/app/backups`
5. Automatically restores PostgreSQL and Weaviate data
6. Application starts with restored data

## Logs

To check if auto-restore ran successfully:

```bash
# View backend container logs
docker logs rag-backend

# Look for these messages:
# "Backups found in /app/backups"
# "Database is empty - attempting auto-restore..."
# "✅ PostgreSQL restored successfully"
# "✅ Weaviate restored successfully"
```

## Manual Restore

If you need to restore backups manually (when database is not empty):

1. Go to `/dashboard/backup`
2. Upload the backup files
3. Click "Restore" for each backup type

## Security Notes

- Backup files may contain sensitive data (user info, course content, etc.)
- Consider encrypting backups before committing to public repositories
- Use private repositories for production backups
- Regularly rotate and clean up old backups

## Troubleshooting

### Auto-restore didn't run

Check if database was already populated:
```bash
docker exec -it rag-backend python -c "
from app.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
result = db.execute(text('SELECT COUNT(*) FROM users')).scalar()
print(f'Users in database: {result}')
db.close()
"
```

### Backup files not found

Verify volume mount:
```bash
docker exec -it rag-backend ls -la /app/backups
```

### Restore failed

Check container logs for detailed error messages:
```bash
docker logs rag-backend 2>&1 | grep -A 10 "auto-restore"
```

## File Structure

```
backups/
├── README.md                          # This file
├── postgres-20260205-143000.sql       # PostgreSQL backup (example)
├── weaviate-20260205-143000.json      # Weaviate backup (example)
└── full-backup-20260205-143000.tar.gz # Full backup archive (optional)
```

## Notes

- The `backups/` directory is no longer in `.gitignore` to allow committing backups
- Only the latest backup of each type is used for auto-restore
- Full backup archives (`.tar.gz`) are not used by auto-restore (extract them first)
- Auto-restore only runs when database is completely empty (first deployment)
