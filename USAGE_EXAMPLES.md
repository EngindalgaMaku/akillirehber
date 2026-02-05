# 💡 Kullanım Örnekleri

## Senaryo 1: İlk Kez Production'a Geçiş

### Durum
- Localhost'ta geliştirme yapıyorsunuz
- 5 kurs, 20 döküman, 100 öğrenci verisi var
- Coolify'da production ortamı hazır

### Çözüm

```bash
# 1. SSH key kurulumu (ilk kez)
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@coolify-server.com

# 2. Tek komutla aktarım
./sync-to-coolify.sh user@coolify-server.com

# 3. Coolify'da restore
ssh user@coolify-server.com
cd ~/rag-backups
docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-*.sql
docker run --rm -v $(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/weaviate-*.tar.gz'
docker compose restart

# 4. Test
curl https://your-api-domain.com/health
```

**Süre:** 10-15 dakika  
**Zorluk:** ⭐ Kolay

---

## Senaryo 2: Haftalık Staging → Production Güncellemesi

### Durum
- Her hafta staging'de test ediyorsunuz
- Testler başarılı olunca production'a aktarıyorsunuz
- Manuel işlem çok zaman alıyor

### Çözüm

```bash
# 1. Staging'de otomatik yedekleme kur (bir kez)
./setup-auto-sync.sh
# Seçenek 1: Localhost (yedek alıcı)

# 2. Production'da otomatik restore kur (bir kez)
ssh user@production-server.com
./setup-auto-sync.sh
# Seçenek 2: Coolify (yedek alıcı)

# 3. Her hafta sadece restore et
ssh user@production-server.com
/usr/local/bin/rag-restore
```

**Süre:** 20 dakika kurulum, sonra 5 dakika/hafta  
**Zorluk:** ⭐⭐ Orta

---

## Senaryo 3: Gerçek Zamanlı Dev → Staging Senkronizasyonu

### Durum
- Ekip olarak çalışıyorsunuz
- Dev ortamındaki değişikliklerin hemen staging'e yansımasını istiyorsunuz
- Manuel senkronizasyon hata yapma riski taşıyor

### Çözüm

```bash
# 1. Dev sunucusunda gerçek zamanlı senkronizasyon kur
./setup-auto-sync.sh
# Seçenek 3: İki yönlü senkronizasyon
# Hedef: staging-server.com

# 2. Lsyncd otomatik çalışacak
sudo systemctl status lsyncd

# 3. Logları izle
tail -f /var/log/lsyncd/lsyncd.log
```

**Süre:** 15 dakika kurulum, sonra otomatik  
**Zorluk:** ⭐⭐⭐ İleri

---

## Senaryo 4: Disaster Recovery (Felaket Kurtarma)

### Durum
- Production sunucunuz çöktü
- Son yedeği hızlıca restore etmeniz gerekiyor
- Veri kaybı kabul edilemez

### Çözüm

```bash
# ÖNCELİKLE: Günlük otomatik yedekleme kurulu olmalı
# (Senaryo 2'deki adımları takip edin)

# Felaket anında:

# 1. Yeni sunucu hazırlayın
ssh user@new-production-server.com

# 2. Docker ve Coolify kurun
# (Coolify dokümantasyonunu takip edin)

# 3. En son yedeği restore edin
cd ~/rag-backups
ls -lt  # En son yedeği bulun

docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-YYYYMMDD-HHMMSS.sql
docker run --rm -v $(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/weaviate-YYYYMMDD-HHMMSS.tar.gz'
docker compose restart

# 4. DNS'i yeni sunucuya yönlendirin
# 5. SSL sertifikasını yenileyin
```

**Süre:** 30-60 dakika (sunucu hazırlığına bağlı)  
**Zorluk:** ⭐⭐⭐ İleri

---

## Senaryo 5: Multi-Region Deployment

### Durum
- Farklı bölgelerde sunucularınız var (EU, US, Asia)
- Her bölgede aynı veriler olmalı
- Düşük latency için local veri gerekli

### Çözüm

```bash
# 1. Ana sunucuda (EU) otomatik yedekleme
./setup-auto-sync.sh
# Seçenek 1: Localhost (yedek alıcı)

# 2. Her bölgede otomatik restore
# US sunucusunda:
ssh user@us-server.com
./setup-auto-sync.sh
# Seçenek 2: Coolify (yedek alıcı)
# Kaynak: eu-server.com

# Asia sunucusunda:
ssh user@asia-server.com
./setup-auto-sync.sh
# Seçenek 2: Coolify (yedek alıcı)
# Kaynak: eu-server.com

# 3. Load balancer ile bölge bazlı yönlendirme
# (Cloudflare, AWS Route53, vb.)
```

**Süre:** 1 saat kurulum, sonra otomatik  
**Zorluk:** ⭐⭐⭐⭐ Uzman

---

## Senaryo 6: Development → Staging → Production Pipeline

### Durum
- 3 aşamalı deployment pipeline'ınız var
- Her aşamada test ve onay gerekiyor
- Otomatik ama kontrollü bir süreç istiyorsunuz

### Çözüm

```bash
# 1. Development'ta günlük yedekleme
# dev-server.com
./setup-auto-sync.sh
# Seçenek 1: Localhost (yedek alıcı)
# Cron: Her gece 02:00

# 2. Staging'e manuel push (test için)
# staging-server.com
cat > /usr/local/bin/pull-from-dev << 'EOF'
#!/bin/bash
rsync -avz dev-server.com:/var/backups/rag/ /var/backups/rag/
/usr/local/bin/rag-restore
EOF
chmod +x /usr/local/bin/pull-from-dev

# 3. Production'a onaylı push (haftalık)
# production-server.com
cat > /usr/local/bin/pull-from-staging << 'EOF'
#!/bin/bash
echo "Staging'den production'a aktarım yapılacak!"
read -p "Onaylıyor musunuz? (yes/no): " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    rsync -avz staging-server.com:/var/backups/rag/ /var/backups/rag/
    /usr/local/bin/rag-restore
    echo "✅ Production güncellendi!"
else
    echo "❌ İşlem iptal edildi"
fi
EOF
chmod +x /usr/local/bin/pull-from-staging

# Kullanım:
# Staging'e push: ssh staging-server.com /usr/local/bin/pull-from-dev
# Production'a push: ssh production-server.com /usr/local/bin/pull-from-staging
```

**Süre:** 2 saat kurulum, sonra 10 dakika/hafta  
**Zorluk:** ⭐⭐⭐⭐ Uzman

---

## Senaryo 7: Selective Sync (Seçici Senkronizasyon)

### Durum
- Sadece belirli kursları production'a aktarmak istiyorsunuz
- Tüm veritabanını aktarmak gereksiz
- Özel filtreleme gerekiyor

### Çözüm

```bash
# 1. Özel yedekleme scripti oluşturun
cat > selective-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)
COURSE_IDS="1,2,3"  # Aktarılacak kurs ID'leri

# Sadece belirli kursları yedekle
docker exec rag-postgres pg_dump -U raguser -d ragchatbot \
    --table=courses \
    --table=documents \
    --table=users \
    --where="course_id IN ($COURSE_IDS)" \
    > $BACKUP_DIR/selective-$DATE.sql

# Weaviate'de sadece bu kursların vektörlerini yedekle
# (Weaviate API kullanarak filtreleme)
curl -X POST http://localhost:8080/v1/backups/filesystem \
  -H "Content-Type: application/json" \
  -d "{
    \"id\": \"selective-$DATE\",
    \"include\": [\"Course_$COURSE_IDS\"]
  }"

echo "Seçici yedekleme tamamlandı: $DATE"
EOF

chmod +x selective-backup.sh
./selective-backup.sh
```

**Süre:** 30 dakika kurulum, 5 dakika/aktarım  
**Zorluk:** ⭐⭐⭐⭐ Uzman

---

## Senaryo 8: Rollback (Geri Alma)

### Durum
- Production'da bir güncelleme yaptınız
- Sorun çıktı, eski versiyona dönmek istiyorsunuz
- Hızlı rollback gerekiyor

### Çözüm

```bash
# 1. Yedekleri versiyonlayın
# Her yedekte git commit hash'i veya versiyon numarası ekleyin
cat > versioned-backup.sh << 'EOF'
#!/bin/bash
VERSION=$(git rev-parse --short HEAD)
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="./backups/v$VERSION-$DATE"

mkdir -p $BACKUP_DIR

docker exec rag-postgres pg_dump -U raguser -d ragchatbot > $BACKUP_DIR/postgres.sql
docker run --rm -v rag-weaviate-data:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/weaviate.tar.gz /data

echo "$VERSION" > $BACKUP_DIR/VERSION
echo "$DATE" > $BACKUP_DIR/TIMESTAMP

echo "Versiyonlu yedek: v$VERSION-$DATE"
EOF

# 2. Rollback scripti
cat > rollback.sh << 'EOF'
#!/bin/bash
echo "Mevcut yedekler:"
ls -lt backups/

read -p "Hangi versiyona dönmek istiyorsunuz? (örn: v1a2b3c4-20260205-120000): " VERSION

if [ -d "backups/$VERSION" ]; then
    echo "Rollback yapılıyor: $VERSION"
    
    docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < backups/$VERSION/postgres.sql
    docker run --rm -v $(docker volume ls -qf name=weaviate):/data -v $(pwd)/backups/$VERSION:/backup alpine sh -c 'cd / && tar xzf /backup/weaviate.tar.gz'
    docker compose restart
    
    echo "✅ Rollback tamamlandı: $VERSION"
else
    echo "❌ Versiyon bulunamadı: $VERSION"
fi
EOF

chmod +x versioned-backup.sh rollback.sh

# Kullanım:
# Yedek al: ./versioned-backup.sh
# Geri al: ./rollback.sh
```

**Süre:** 5-10 dakika  
**Zorluk:** ⭐⭐⭐ İleri

---

## 🎯 Hangi Senaryoyu Seçmeliyim?

| Durum | Önerilen Senaryo | Zorluk | Süre |
|-------|------------------|--------|------|
| İlk production deployment | Senaryo 1 | ⭐ | 15 dk |
| Düzenli güncellemeler | Senaryo 2 | ⭐⭐ | 20 dk kurulum |
| Ekip çalışması | Senaryo 3 | ⭐⭐⭐ | 15 dk kurulum |
| Yedekleme stratejisi | Senaryo 4 | ⭐⭐⭐ | 30 dk kurulum |
| Global deployment | Senaryo 5 | ⭐⭐⭐⭐ | 1 saat kurulum |
| CI/CD pipeline | Senaryo 6 | ⭐⭐⭐⭐ | 2 saat kurulum |
| Özel gereksinimler | Senaryo 7 | ⭐⭐⭐⭐ | 30 dk kurulum |
| Hata durumu | Senaryo 8 | ⭐⭐⭐ | 5 dk |

---

## 💡 İpuçları

### Performans
- Sıkıştırma kullanın (gzip, tar.gz)
- Büyük dosyalar için rsync kullanın (incremental)
- Paralel aktarım için GNU parallel kullanın

### Güvenlik
- SSH key authentication kullanın
- Yedekleri şifreleyin (GPG)
- Firewall kurallarını yapılandırın
- VPN kullanın (hassas veriler için)

### Monitoring
- Yedekleme loglarını izleyin
- Disk alanını kontrol edin
- Başarısız yedeklemeler için alert kurun
- Restore testleri yapın (düzenli)

### Otomasyon
- Cron job kullanın (zamanlanmış görevler)
- Systemd service kullanın (sürekli çalışan servisler)
- Webhook kullanın (event-driven)
- CI/CD pipeline entegrasyonu

---

## 📞 Yardım

Daha fazla bilgi için:
- [Hızlı Başlangıç](./SYNC_QUICKSTART.md)
- [Detaylı Rehber](./COOLIFY_SYNC_GUIDE.md)
- [Özet](./SYNC_SUMMARY.md)
