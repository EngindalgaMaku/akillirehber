#!/bin/bash
# Localhost verilerini Coolify ortamına aktarma scripti
# Kullanım: ./sync-to-coolify.sh [coolify-host]

set -e

# Renkli çıktı için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Coolify host bilgisi
COOLIFY_HOST="${1:-}"
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)

echo -e "${BLUE}=========================================="
echo "Coolify Veri Senkronizasyon Aracı"
echo -e "==========================================${NC}\n"

# Coolify host kontrolü
if [ -z "$COOLIFY_HOST" ]; then
    echo -e "${RED}❌ HATA: Coolify host adresi belirtilmedi!${NC}"
    echo -e "${YELLOW}Kullanım: ./sync-to-coolify.sh user@coolify-server.com${NC}"
    echo -e "${YELLOW}Örnek: ./sync-to-coolify.sh root@192.168.1.100${NC}\n"
    exit 1
fi

echo -e "${GREEN}✅ Hedef sunucu: $COOLIFY_HOST${NC}\n"

# Yedek dizini oluştur
mkdir -p $BACKUP_DIR

# ===========================================
# 1. PostgreSQL Veritabanı Yedeği
# ===========================================
echo -e "${BLUE}[1/3] PostgreSQL veritabanı yedekleniyor...${NC}"

# Container çalışıyor mu kontrol et
if ! docker ps | grep -q "rag-postgres"; then
    echo -e "${RED}❌ PostgreSQL container'ı çalışmıyor!${NC}"
    exit 1
fi

# PostgreSQL dump al
docker exec rag-postgres pg_dump -U raguser -d ragchatbot > "$BACKUP_DIR/postgres-$DATE.sql"

if [ -f "$BACKUP_DIR/postgres-$DATE.sql" ]; then
    PG_SIZE=$(du -h "$BACKUP_DIR/postgres-$DATE.sql" | cut -f1)
    echo -e "${GREEN}✅ PostgreSQL yedeği alındı: $PG_SIZE${NC}\n"
else
    echo -e "${RED}❌ PostgreSQL yedeği alınamadı!${NC}"
    exit 1
fi

# ===========================================
# 2. Weaviate Vektör Veritabanı Yedeği
# ===========================================
echo -e "${BLUE}[2/3] Weaviate vektör veritabanı yedekleniyor...${NC}"

# Container çalışıyor mu kontrol et
if ! docker ps | grep -q "rag-weaviate"; then
    echo -e "${YELLOW}⚠️  Weaviate container'ı çalışmıyor!${NC}"
else
    # Weaviate volume'unu yedekle
    docker run --rm \
        -v rag-weaviate-data:/data \
        -v $(pwd)/$BACKUP_DIR:/backup \
        alpine tar czf /backup/weaviate-$DATE.tar.gz /data

    if [ -f "$BACKUP_DIR/weaviate-$DATE.tar.gz" ]; then
        WV_SIZE=$(du -h "$BACKUP_DIR/weaviate-$DATE.tar.gz" | cut -f1)
        echo -e "${GREEN}✅ Weaviate yedeği alındı: $WV_SIZE${NC}\n"
    else
        echo -e "${RED}❌ Weaviate yedeği alınamadı!${NC}"
        exit 1
    fi
fi

# ===========================================
# 3. Yedekleri Coolify'a Aktar
# ===========================================
echo -e "${BLUE}[3/3] Yedekler Coolify sunucusuna aktarılıyor...${NC}"

# SSH bağlantısını test et
if ! ssh -o ConnectTimeout=5 "$COOLIFY_HOST" "echo 'SSH bağlantısı başarılı'" > /dev/null 2>&1; then
    echo -e "${RED}❌ SSH bağlantısı kurulamadı!${NC}"
    echo -e "${YELLOW}Lütfen SSH anahtarınızın eklendiğinden emin olun:${NC}"
    echo -e "${YELLOW}  ssh-copy-id $COOLIFY_HOST${NC}\n"
    exit 1
fi

# Coolify sunucusunda yedek dizini oluştur
ssh "$COOLIFY_HOST" "mkdir -p ~/rag-backups"

# PostgreSQL yedeğini aktar
echo -e "${YELLOW}PostgreSQL yedeği aktarılıyor...${NC}"
scp "$BACKUP_DIR/postgres-$DATE.sql" "$COOLIFY_HOST:~/rag-backups/"
echo -e "${GREEN}✅ PostgreSQL yedeği aktarıldı${NC}"

# Weaviate yedeğini aktar
echo -e "${YELLOW}Weaviate yedeği aktarılıyor...${NC}"
scp "$BACKUP_DIR/weaviate-$DATE.tar.gz" "$COOLIFY_HOST:~/rag-backups/"
echo -e "${GREEN}✅ Weaviate yedeği aktarıldı${NC}\n"

# ===========================================
# 4. Restore Talimatları
# ===========================================
echo -e "${BLUE}=========================================="
echo "Yedekler Başarıyla Aktarıldı!"
echo -e "==========================================${NC}\n"

echo -e "${YELLOW}Coolify sunucusunda restore işlemi için:${NC}\n"

echo -e "${GREEN}1. Coolify sunucusuna bağlanın:${NC}"
echo -e "   ssh $COOLIFY_HOST\n"

echo -e "${GREEN}2. PostgreSQL'i restore edin:${NC}"
echo -e "   cd ~/rag-backups"
echo -e "   docker exec -i \$(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-$DATE.sql\n"

echo -e "${GREEN}3. Weaviate'i restore edin:${NC}"
echo -e "   docker run --rm -v \$(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/weaviate-$DATE.tar.gz'\n"

echo -e "${GREEN}4. Container'ları yeniden başlatın:${NC}"
echo -e "   docker compose restart\n"

echo -e "${BLUE}=========================================="
echo "Otomatik restore için:"
echo -e "==========================================${NC}"
echo -e "${YELLOW}./restore-on-coolify.sh $COOLIFY_HOST postgres-$DATE.sql weaviate-$DATE.tar.gz${NC}\n"

# Restore script'ini oluştur
cat > restore-on-coolify.sh << 'RESTORE_SCRIPT'
#!/bin/bash
# Coolify sunucusunda restore işlemi
# Kullanım: ./restore-on-coolify.sh [coolify-host] [postgres-backup] [weaviate-backup]

set -e

COOLIFY_HOST="${1:-}"
PG_BACKUP="${2:-}"
WV_BACKUP="${3:-}"

if [ -z "$COOLIFY_HOST" ] || [ -z "$PG_BACKUP" ] || [ -z "$WV_BACKUP" ]; then
    echo "Kullanım: ./restore-on-coolify.sh user@host postgres-backup.sql weaviate-backup.tar.gz"
    exit 1
fi

echo "Coolify sunucusunda restore işlemi başlatılıyor..."

ssh "$COOLIFY_HOST" << EOF
cd ~/rag-backups

echo "PostgreSQL restore ediliyor..."
docker exec -i \$(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < $PG_BACKUP

echo "Weaviate restore ediliyor..."
docker run --rm -v \$(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/$WV_BACKUP'

echo "Container'lar yeniden başlatılıyor..."
cd ~/your-project-path
docker compose restart

echo "✅ Restore işlemi tamamlandı!"
EOF
RESTORE_SCRIPT

chmod +x restore-on-coolify.sh

echo -e "${GREEN}✅ restore-on-coolify.sh scripti oluşturuldu${NC}\n"
