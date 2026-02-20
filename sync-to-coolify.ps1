# Localhost verilerini Coolify ortamına aktarma scripti (Windows PowerShell)
# Kullanım: .\sync-to-coolify.ps1 -CoolifyHost "user@coolify-server.com"

param(
    [Parameter(Mandatory=$true)]
    [string]$CoolifyHost
)

$ErrorActionPreference = "Stop"

# Renkli çıktı için
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

$BackupDir = ".\backups"
$Date = Get-Date -Format "yyyyMMdd-HHmmss"

Write-ColorOutput Blue "=========================================="
Write-ColorOutput Blue "Coolify Veri Senkronizasyon Aracı"
Write-ColorOutput Blue "=========================================="
Write-Output ""

Write-ColorOutput Green "✅ Hedef sunucu: $CoolifyHost"
Write-Output ""

# Yedek dizini oluştur
if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir | Out-Null
}

# ===========================================
# 1. PostgreSQL Veritabanı Yedeği
# ===========================================
Write-ColorOutput Blue "[1/3] PostgreSQL veritabanı yedekleniyor..."

# Container çalışıyor mu kontrol et
$pgContainer = docker ps --filter "name=rag-postgres" --format "{{.Names}}"
if (-not $pgContainer) {
    Write-ColorOutput Red "❌ PostgreSQL container'ı çalışmıyor!"
    exit 1
}

# PostgreSQL dump al
$pgBackupFile = "$BackupDir\postgres-$Date.sql"
docker exec rag-postgres pg_dump -U raguser -d ragchatbot | Out-File -FilePath $pgBackupFile -Encoding UTF8

if (Test-Path $pgBackupFile) {
    $pgSize = (Get-Item $pgBackupFile).Length / 1MB
    Write-ColorOutput Green "✅ PostgreSQL yedeği alındı: $([math]::Round($pgSize, 2)) MB"
    Write-Output ""
} else {
    Write-ColorOutput Red "❌ PostgreSQL yedeği alınamadı!"
    exit 1
}

# ===========================================
# 2. Weaviate Vektör Veritabanı Yedeği
# ===========================================
Write-ColorOutput Blue "[2/3] Weaviate vektör veritabanı yedekleniyor..."

# Container çalışıyor mu kontrol et
$wvContainer = docker ps --filter "name=rag-weaviate" --format "{{.Names}}"
if (-not $wvContainer) {
    Write-ColorOutput Yellow "⚠️  Weaviate container'ı çalışmıyor!"
} else {
    # Weaviate volume'unu yedekle
    $wvBackupFile = "weaviate-$Date.tar.gz"
    docker run --rm `
        -v rag-weaviate-data:/data `
        -v ${PWD}\${BackupDir}:/backup `
        alpine tar czf /backup/$wvBackupFile /data

    if (Test-Path "$BackupDir\$wvBackupFile") {
        $wvSize = (Get-Item "$BackupDir\$wvBackupFile").Length / 1MB
        Write-ColorOutput Green "✅ Weaviate yedeği alındı: $([math]::Round($wvSize, 2)) MB"
        Write-Output ""
    } else {
        Write-ColorOutput Red "❌ Weaviate yedeği alınamadı!"
        exit 1
    }
}

# ===========================================
# 3. Yedekleri Coolify'a Aktar
# ===========================================
Write-ColorOutput Blue "[3/3] Yedekler Coolify sunucusuna aktarılıyor..."

# SSH bağlantısını test et (scp kullanılabilirliğini kontrol et)
Write-ColorOutput Yellow "SSH bağlantısı test ediliyor..."

# SCP ile dosyaları aktar
Write-ColorOutput Yellow "PostgreSQL yedeği aktarılıyor..."
scp $pgBackupFile "${CoolifyHost}:~/rag-backups/"
if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✅ PostgreSQL yedeği aktarıldı"
} else {
    Write-ColorOutput Red "❌ PostgreSQL yedeği aktarılamadı!"
    Write-ColorOutput Yellow "Lütfen SSH anahtarınızın eklendiğinden emin olun:"
    Write-ColorOutput Yellow "  ssh-copy-id $CoolifyHost"
    exit 1
}

Write-ColorOutput Yellow "Weaviate yedeği aktarılıyor..."
scp "$BackupDir\$wvBackupFile" "${CoolifyHost}:~/rag-backups/"
if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✅ Weaviate yedeği aktarıldı"
} else {
    Write-ColorOutput Red "❌ Weaviate yedeği aktarılamadı!"
    exit 1
}

Write-Output ""

# ===========================================
# 4. Restore Talimatları
# ===========================================
Write-ColorOutput Blue "=========================================="
Write-ColorOutput Blue "Yedekler Başarıyla Aktarıldı!"
Write-ColorOutput Blue "=========================================="
Write-Output ""

Write-ColorOutput Yellow "Coolify sunucusunda restore işlemi için:"
Write-Output ""

Write-ColorOutput Green "1. Coolify sunucusuna bağlanın:"
Write-Output "   ssh $CoolifyHost"
Write-Output ""

Write-ColorOutput Green "2. PostgreSQL'i restore edin:"
Write-Output "   cd ~/rag-backups"
Write-Output "   docker exec -i `$(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-$Date.sql"
Write-Output ""

Write-ColorOutput Green "3. Weaviate'i restore edin:"
Write-Output "   docker run --rm -v `$(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/$wvBackupFile'"
Write-Output ""

Write-ColorOutput Green "4. Container'ları yeniden başlatın:"
Write-Output "   docker compose restart"
Write-Output ""

Write-ColorOutput Blue "=========================================="
Write-ColorOutput Blue "Yedekleme Tamamlandı!"
Write-ColorOutput Blue "=========================================="
Write-Output ""

Write-ColorOutput Yellow "Yerel yedek dosyaları:"
Write-Output "  - $pgBackupFile"
Write-Output "  - $BackupDir\$wvBackupFile"
Write-Output ""
