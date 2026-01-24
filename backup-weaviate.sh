#!/bin/bash
# Weaviate Otomatik Yedekleme Script'i
# Kullanım: ./backup-weaviate.sh

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)

# Yedek dizini oluştur
mkdir -p $BACKUP_DIR

echo "=========================================="
echo "Weaviate Yedekleme Başlatılıyor..."
echo "=========================================="
echo ""

# Weaviate container'ının durumunu kontrol et
CONTAINER_STATUS=$(docker ps --filter "name=rag-weaviate" --format "{{.Status}}")

if [ -z "$CONTAINER_STATUS" ]; then
    echo "⚠️  UYARI: Weaviate container'ı çalışmıyor!"
    echo "Yine de yedek alınıyor ama container çalışmıyorsa veriler güncel olmayabilir."
else
    echo "✅ Weaviate container'ı çalışıyor: $CONTAINER_STATUS"
fi

echo ""
echo "Yedek alınıyor..."
docker run --rm \
    -v rag-weaviate-data:/data \
    -v $(pwd)/$BACKUP_DIR:/backup \
    alpine tar czf /backup/weaviate-$DATE.tar.gz /data

# Yedek dosyasının boyutunu kontrol et
BACKUP_FILE="$BACKUP_DIR/weaviate-$DATE.tar.gz"
if [ -f "$BACKUP_FILE" ]; then
    FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "✅ Yedek başarıyla tamamlandı!"
    echo "📁 Dosya: $BACKUP_FILE"
    echo "📊 Boyut: $FILE_SIZE"

    # Son 7 yedeği tut, diğerlerini sil
    echo ""
    echo "Eski yedekler temizleniyor (son 7 gün)..."
    find $BACKUP_DIR -name "weaviate-*.tar.gz" -mtime +7 -delete

    # Kalan yedekleri listele
    echo ""
    echo "Mevcut yedekler:"
    ls -lh $BACKUP_DIR/weaviate-*.tar.gz 2>/dev/null || echo "  (Yedek bulunamadı)"
else
    echo "❌ HATA: Yedek dosyası oluşturulamadı!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Yedekleme Tamamlandı!"
echo "=========================================="
