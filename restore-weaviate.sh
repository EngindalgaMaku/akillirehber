#!/bin/bash
# Weaviate Kurtarma Script'i
# Kullanım: ./restore-weaviate.sh <yedek-dosyası>

if [ -z "$1" ]; then
    echo "❌ HATA: Yedek dosyası belirtilmedi!"
    echo ""
    echo "Kullanım: ./restore-weaviate.sh <yedek-dosyası>"
    echo ""
    echo "Mevcut yedekler:"
    ls -lh backups/weaviate-*.tar.gz 2>/dev/null || echo "  (Yedek bulunamadı)"
    exit 1
fi

BACKUP_FILE="backups/$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ HATA: Yedek dosyası bulunamadı: $BACKUP_FILE"
    exit 1
fi

echo "=========================================="
echo "Weaviate Kurtarma İşlemi Başlatılıyor..."
echo "=========================================="
echo ""
echo "📁 Yedek dosyası: $BACKUP_FILE"
FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "📊 Boyut: $FILE_SIZE"
echo ""

# Weaviate container'ının durumunu kontrol et
CONTAINER_STATUS=$(docker ps --filter "name=rag-weaviate" --format "{{.Status}}")

if [ -n "$CONTAINER_STATUS" ]; then
    echo "⚠️  Weaviate container'ı durduruluyor..."
    docker-compose stop weaviate
    echo "✅ Container durduruldu"
else
    echo "ℹ️  Weaviate container'ı zaten durdurulmuş"
fi

echo ""
echo "⚠️  DİKKAT: Mevcut veriler silinecek!"
echo "Devam etmek için 'e' tuşuna basın, iptal için 'Ctrl+C'..."
read -p "Devam etmek istiyor musunuz? (e/h): " confirm

if [ "$confirm" != "e" ] && [ "$confirm" != "E" ]; then
    echo "❌ İşlem iptal edildi."
    exit 0
fi

echo ""
echo "Mevcut volume siliniyor..."
docker volume rm rag-weaviate-data
echo "✅ Volume silindi"

echo ""
echo "Yeni volume oluşturuluyor..."
docker volume create rag-weaviate-data
echo "✅ Volume oluşturuldu"

echo ""
echo "Yedek geri yükleniyor..."
docker run --rm \
    -v rag-weaviate-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar xzf /backup/$1 -C /data

echo "✅ Yedek başarıyla geri yüklendi"

echo ""
echo "İzinler düzeltiliyor..."
docker run --rm \
    -v rag-weaviate-data:/data \
    alpine chown -R 1000:1000 /data

echo "✅ İzinler düzeltildi"

echo ""
echo "=========================================="
echo "Kurtarma Tamamlandı!"
echo "=========================================="
echo ""
echo "Weaviate container'ı başlatmak için:"
echo "  docker-compose up -d weaviate"
echo ""
echo "Logları izlemek için:"
echo "  docker logs -f rag-weaviate"
