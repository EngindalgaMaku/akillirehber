#!/bin/bash
# Weaviate Sürüm Güncelleme Script'i
# Kullanım: ./upgrade-weaviate.sh <yeni-sürüm>

if [ -z "$1" ]; then
    echo "❌ HATA: Yeni sürüm belirtilmedi!"
    echo ""
    echo "Kullanım: ./upgrade-weaviate.sh <yeni-sürüm>"
    echo ""
    echo "Örnek: ./upgrade-weaviate.sh 1.35.3"
    echo ""
    echo "Mevcut sürüm kontrol ediliyor..."
    docker ps --filter "name=rag-weaviate" --format "table {{.Names}}\t{{.Image}}"
    exit 1
fi

NEW_VERSION=$1

echo "=========================================="
echo "Weaviate Sürüm Güncellemesi"
echo "=========================================="
echo ""
echo "📦 Yeni sürüm: $NEW_VERSION"
echo ""

# Mevcut sürümü kontrol et
CURRENT_IMAGE=$(docker ps --filter "name=rag-weaviate" --format "{{.Image}}")
echo "📦 Mevcut sürüm: $CURRENT_IMAGE"
echo ""

# Yedek al
echo "=========================================="
echo "ADIM 1: Yedek Alma"
echo "=========================================="
./backup-weaviate.sh

if [ $? -ne 0 ]; then
    echo "❌ HATA: Yedekleme başarısız! Güncelleme iptal ediliyor."
    exit 1
fi

echo ""
echo "=========================================="
echo "ADIM 2: Sürüm Güncelleme"
echo "=========================================="
echo ""
echo "docker-compose.yml dosyası güncelleniyor..."

# docker-compose.yml dosyasını güncelle
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    sed -i "s|semitechnologies/weaviate:[0-9.]*|semitechnologies/weaviate:$NEW_VERSION|g" docker-compose.yml
else
    # Linux/Mac
    sed -i.bak "s|semitechnologies/weaviate:[0-9.]*|semitechnologies/weaviate:$NEW_VERSION|g" docker-compose.yml
fi

echo "✅ docker-compose.yml güncellendi"

echo ""
echo "=========================================="
echo "ADIM 3: Container Yeniden Başlatma"
echo "=========================================="
echo ""
echo "Weaviate container'ı yeniden başlatılıyor..."
docker-compose up -d weaviate

echo "✅ Container başlatıldı"

echo ""
echo "=========================================="
echo "ADIM 4: Durum Kontrolü"
echo "=========================================="
echo ""

# Container'ın başlamasını bekle
echo "Container'ın başlaması bekleniyor (30 saniye)..."
sleep 30

# Durumu kontrol et
CONTAINER_STATUS=$(docker ps --filter "name=rag-weaviate" --format "{{.Status}}")
NEW_IMAGE=$(docker ps --filter "name=rag-weaviate" --format "{{.Image}}")

echo "Container durumu: $CONTAINER_STATUS"
echo "Sürüm: $NEW_IMAGE"
echo ""

# Logları kontrol et
echo "Son 20 satır log:"
docker logs --tail 20 rag-weaviate

echo ""
echo "=========================================="
echo "Güncelleme Tamamlandı!"
echo "=========================================="
echo ""
echo "✅ Weaviate $NEW_VERSION sürümüne güncellendi!"
echo "✅ Yedek alındı: backups/weaviate-backup-*.tar.gz"
echo ""
echo "Sorun olursa, kurtarma için:"
echo "  ./restore-weaviate.sh <yedek-dosyası>"
echo ""
echo "Logları izlemek için:"
echo "  docker logs -f rag-weaviate"
