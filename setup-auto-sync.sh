#!/bin/bash
# Otomatik senkronizasyon kurulum scripti
# Bu script hem localhost hem de Coolify sunucusunda çalıştırılabilir

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "Otomatik Senkronizasyon Kurulumu"
echo -e "==========================================${NC}\n"

# Kurulum tipi seçimi
echo -e "${YELLOW}Kurulum tipini seçin:${NC}"
echo "1) Localhost (yedek alıcı)"
echo "2) Coolify Server (yedek alıcı)"
echo "3) İki yönlü senkronizasyon"
read -p "Seçiminiz (1-3): " SETUP_TYPE

case $SETUP_TYPE in
    1)
        echo -e "\n${GREEN}Localhost için yedekleme kurulumu başlatılıyor...${NC}\n"
        
        # Backup script oluştur
        cat > /usr/local/bin/rag-backup << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/rag"
DATE=$(date +%Y%m%d-%H%M%S)
mkdir -p $BACKUP_DIR

# PostgreSQL
docker exec rag-postgres pg_dump -U raguser -d ragchatbot | gzip > $BACKUP_DIR/postgres-$DATE.sql.gz

# Weaviate
docker run --rm \
    -v rag-weaviate-data:/data \
    -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/weaviate-$DATE.tar.gz /data

# Eski yedekleri temizle (7 günden eski)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "Yedekleme tamamlandı: $DATE"
EOF

        chmod +x /usr/local/bin/rag-backup
        
        # Cron job ekle
        (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/rag-backup") | crontab -
        
        echo -e "${GREEN}✅ Localhost yedekleme kuruldu${NC}"
        echo -e "${YELLOW}Yedekler her gece saat 02:00'de alınacak${NC}"
        echo -e "${YELLOW}Yedek konumu: /var/backups/rag${NC}\n"
        ;;
        
    2)
        echo -e "\n${GREEN}Coolify Server için restore kurulumu başlatılıyor...${NC}\n"
        
        read -p "Localhost IP adresi: " LOCALHOST_IP
        read -p "Localhost kullanıcı adı: " LOCALHOST_USER
        
        # Restore script oluştur
        cat > /usr/local/bin/rag-restore << EOF
#!/bin/bash
BACKUP_DIR="/var/backups/rag"
REMOTE_BACKUP="$LOCALHOST_USER@$LOCALHOST_IP:/var/backups/rag"

# En son yedeği çek
rsync -avz --progress \$REMOTE_BACKUP/ \$BACKUP_DIR/

# En son dosyaları bul
LATEST_PG=\$(ls -t \$BACKUP_DIR/postgres-*.sql.gz | head -1)
LATEST_WV=\$(ls -t \$BACKUP_DIR/weaviate-*.tar.gz | head -1)

if [ -z "\$LATEST_PG" ] || [ -z "\$LATEST_WV" ]; then
    echo "Yedek dosyaları bulunamadı!"
    exit 1
fi

echo "PostgreSQL restore ediliyor: \$LATEST_PG"
gunzip -c \$LATEST_PG | docker exec -i \$(docker ps -qf name=postgres) psql -U raguser -d ragchatbot

echo "Weaviate restore ediliyor: \$LATEST_WV"
docker run --rm -v \$(docker volume ls -qf name=weaviate):/data -v \$BACKUP_DIR:/backup alpine sh -c "cd / && tar xzf /backup/\$(basename \$LATEST_WV)"

echo "Container'lar yeniden başlatılıyor..."
docker compose restart

echo "Restore tamamlandı!"
EOF

        chmod +x /usr/local/bin/rag-restore
        
        # SSH key kurulumu
        echo -e "\n${YELLOW}SSH key kurulumu yapılıyor...${NC}"
        if [ ! -f ~/.ssh/id_rsa ]; then
            ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
        fi
        
        echo -e "\n${YELLOW}Bu public key'i localhost'a ekleyin:${NC}"
        cat ~/.ssh/id_rsa.pub
        echo ""
        
        read -p "SSH key'i eklediniz mi? (y/n): " SSH_READY
        if [ "$SSH_READY" = "y" ]; then
            # Test bağlantısı
            if ssh -o ConnectTimeout=5 $LOCALHOST_USER@$LOCALHOST_IP "echo 'Bağlantı başarılı'"; then
                echo -e "${GREEN}✅ SSH bağlantısı başarılı${NC}"
            else
                echo -e "${RED}❌ SSH bağlantısı başarısız${NC}"
                exit 1
            fi
        fi
        
        # Günlük otomatik restore (opsiyonel)
        read -p "Günlük otomatik restore kurmak ister misiniz? (y/n): " AUTO_RESTORE
        if [ "$AUTO_RESTORE" = "y" ]; then
            (crontab -l 2>/dev/null; echo "0 3 * * * /usr/local/bin/rag-restore") | crontab -
            echo -e "${GREEN}✅ Otomatik restore kuruldu (her gece 03:00)${NC}"
        fi
        
        echo -e "\n${GREEN}✅ Coolify Server restore kuruldu${NC}"
        echo -e "${YELLOW}Manuel restore için: /usr/local/bin/rag-restore${NC}\n"
        ;;
        
    3)
        echo -e "\n${GREEN}İki yönlü senkronizasyon kurulumu başlatılıyor...${NC}\n"
        
        read -p "Uzak sunucu adresi (user@host): " REMOTE_HOST
        
        # Lsyncd kurulumu
        if ! command -v lsyncd &> /dev/null; then
            echo -e "${YELLOW}lsyncd kuruluyor...${NC}"
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y lsyncd
            elif command -v yum &> /dev/null; then
                sudo yum install -y lsyncd
            else
                echo -e "${RED}lsyncd otomatik kurulamadı. Manuel kurulum gerekli.${NC}"
                exit 1
            fi
        fi
        
        # Lsyncd konfigürasyonu
        cat > /etc/lsyncd/lsyncd.conf.lua << EOF
settings {
    logfile = "/var/log/lsyncd/lsyncd.log",
    statusFile = "/var/log/lsyncd/lsyncd.status",
    statusInterval = 20,
    maxProcesses = 4
}

sync {
    default.rsync,
    source = "/var/backups/rag",
    target = "$REMOTE_HOST:/var/backups/rag",
    delay = 300,
    rsync = {
        archive = true,
        compress = true,
        verbose = true
    }
}
EOF

        # Lsyncd servisini başlat
        sudo systemctl enable lsyncd
        sudo systemctl start lsyncd
        
        echo -e "${GREEN}✅ İki yönlü senkronizasyon kuruldu${NC}"
        echo -e "${YELLOW}Senkronizasyon her 5 dakikada bir çalışacak${NC}"
        echo -e "${YELLOW}Log dosyası: /var/log/lsyncd/lsyncd.log${NC}\n"
        ;;
        
    *)
        echo -e "${RED}Geçersiz seçim!${NC}"
        exit 1
        ;;
esac

echo -e "${BLUE}=========================================="
echo "Kurulum Tamamlandı!"
echo -e "==========================================${NC}\n"

# Test komutu
echo -e "${YELLOW}Test için:${NC}"
case $SETUP_TYPE in
    1) echo "  /usr/local/bin/rag-backup" ;;
    2) echo "  /usr/local/bin/rag-restore" ;;
    3) echo "  sudo systemctl status lsyncd" ;;
esac
echo ""
