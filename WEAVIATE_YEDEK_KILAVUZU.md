# Weaviate Yedekleme, Kurtarma ve Sürüm Güncelleme Kılavuzu

Bu kılavuz, Weaviate veritabanını yedekleme, kurtarma ve sürüm güncelleme işlemlerini açıklar.

## 📋 İçindekiler

1. [Oluşturulan Dosyalar](#oluşturulan-dosyalar)
2. [Yedekleme](#yedekleme)
3. [Kurtarma](#kurtarma)
4. [Sürüm Güncelleme](#sürüm-güncelleme)
5. [Acil Durumda Kurtarma](#acil-durumda-kurtarma)

---

## 📁 Oluşturulan Dosyalar

### 1. `backup-weaviate.sh`
Otomatik yedekleme script'i.

### 2. `restore-weaviate.sh`
Yedekten kurtarma script'i.

### 3. `upgrade-weaviate.sh`
Sürüm güncelleme script'i.

### 4. `WEAVIATE_YEDEK_KILAVUZU.md` (bu dosya)
Kullanım kılavuzu.

---

## 💾 Yedekleme

### Otomatik Yedekleme

```bash
# Script'i çalıştır
./backup-weaviate.sh
```

**Script şunları yapar:**
- ✅ Weaviate container'ının durumunu kontrol eder
- ✅ Yedeği `backups/` dizinine alır
- ✅ Dosya boyutunu gösterir
- ✅ Son 7 yedeği tutar, eski yedekleri siler
- ✅ Mevcut yedekleri listeler

**Çıktı örneği:**
```
==========================================
Weaviate Yedekleme Başlatılıyor...
==========================================

✅ Weaviate container'ı çalışıyor: Up 3 hours (healthy)

Yedek alınıyor...
✅ Yedek başarıyla tamamlandı!
📁 Dosya: backups/weaviate-20260122-190652.tar.gz
📊 Boyut: 21.1M

Eski yedekler temizleniyor (son 7 gün)...

Mevcut yedekler:
-rw-r--r-- 1 user user 21.1M Jan 22 19:06 weaviate-20260122-190652.tar.gz
```

---

## 🔄 Kurtarma

### Yedekten Geri Yükleme

```bash
# Mevcut yedekleri listele
ls -lh backups/weaviate-*.tar.gz

# Belirli bir yedeği geri yükle
./restore-weaviate.sh weaviate-backup-20260122-190652.tar.gz
```

**Script şunları yapar:**
- ✅ Yedek dosyasının varlığını kontrol eder
- ✅ Weaviate container'ını durdurur
- ✅ Mevcut volume'u siler (onay ister)
- ✅ Yeni volume oluşturur
- ✅ Yedeği geri yükler
- ✅ Dosya izinlerini düzeltir
- ⚠️  Silmeden önce onay ister

**Önemli:**
- ⚠️  Kurtarma işlemi mevcut verileri siler!
- ⚠️  Script onay ister, 'e' tuşuna basın
- ✅  Her zaman önce yedek alınır

---

## 📦 Sürüm Güncelleme

### 1.27.0 → 1.35.3 Güncelleme

```bash
# Sürümü güncelle
./upgrade-weaviate.sh 1.35.3
```

**Script şunları yapar:**
1. ✅ Mevcut sürümü gösterir
2. ✅ Otomatik yedek alır (`backup-weaviate.sh` çağırır)
3. ✅ `docker-compose.yml` dosyasını günceller
4. ✅ Container'ı yeniden başlatır
5. ✅ Yeni sürümü doğrular
6. ✅ Logları gösterir

**Güncelleme adımları:**
```
==========================================
Weaviate Sürüm Güncellemesi
==========================================

📦 Yeni sürüm: 1.35.3
📦 Mevcut sürüm: semitechnologies/weaviate:1.27.0

==========================================
ADIM 1: Yedek Alma
==========================================
[backup-weaviate.sh çalıştırılır...]

==========================================
ADIM 2: Sürüm Güncelleme
==========================================
docker-compose.yml dosyası güncelleniyor...
✅ docker-compose.yml güncellendi

==========================================
ADIM 3: Container Yeniden Başlatma
==========================================
Weaviate container'ı yeniden başlatılıyor...
✅ Container başlatıldı

==========================================
ADIM 4: Durum Kontrolü
==========================================
Container'ın başlaması bekleniyor (30 saniye)...
Container durumu: Up 30 seconds
Sürüm: semitechnologies/weaviate:1.35.3
```

---

## 🚨 Acil Durumda Kurtarma

### Weaviate Container'ı Çalışmıyorsa

```bash
# 1. Mevcut verileri yedekle (gerekirse)
docker run --rm -v rag-weaviate-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/weaviate-emergency-backup.tar.gz /data

# 2. Container'ı durdur
docker-compose stop weaviate

# 3. Volume'u sil
docker volume rm rag-weaviate-data

# 4. Yeni volume oluştur
docker volume create rag-weaviate-data

# 5. Yedeği geri yükle
docker run --rm -v rag-weaviate-data:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/weaviate-backup-YYYYMMDD.tar.gz -C /data

# 6. İzinleri düzelt
docker run --rm -v rag-weaviate-data:/data alpine chown -R 1000:1000 /data

# 7. Container'ı yeniden başlat
docker-compose up -d weaviate

# 8. Logları kontrol et
docker logs -f rag-weaviate
```

---

## 📊 Sürüm Bilgileri

### Mevcut Sürüm: 1.27.0
- ✅ Stabil ve güvenli
- ✅ Docker volume'da saklanıyor
- ✅ PostgreSQL ile uyumlu

### Hedef Sürüm: 1.35.3
- ✅ En son stabil sürüm (Ocak 2025)
- ✅ Performans iyileştirmeleri içerir
- ✅ Güvenlik güncellemeleri içerir
- ⚠️  Büyük sürüm atlaması (1.27 → 1.35)

---

## ⚠️ Önemli Notlar

### Yedekleme Öncesi
1. ✅ Her zaman yedek alın
2. ✅ Yedek dosyasının boyutunu kontrol edin
3. ✅ Yedek dosyasını güvenli bir yerde saklayın

### Sürüm Güncelleme Öncesi
1. ✅ Otomatik yedek alır (script bu işi yapar)
2. ✅ `docker-compose.yml` dosyasının yedeğini alın
3. ✅ Container loglarını kontrol edin

### Kurtarma Sonrası
1. ✅ Verilerin doğru yüklendiğini kontrol edin
2. ✅ Container loglarını izleyin
3. ✅ Sistemdeki vektör aramalarını test edin

---

## 🔧 Sorun Giderme

### Yedekleme Başarısız Olursa

```bash
# Container durumunu kontrol et
docker ps --filter "name=rag-weaviate"

# Container loglarını kontrol et
docker logs rag-weaviate

# Volume durumunu kontrol et
docker volume ls | grep weaviate
```

### Sürüm Güncelleme Sonrası Sorunlar

```bash
# Container'ı yeniden başlat
docker-compose restart weaviate

# Logları izle
docker logs -f rag-weaviate

# Eğer sorun devam ederse, yedeği geri yükle
./restore-weaviate.sh weaviate-backup-YYYYMMDD.tar.gz
```

### Container Başlamıyorsa

```bash
# Container'ı zorla durdur
docker-compose kill weaviate

# Volume'u kontrol et
docker volume inspect rag-weaviate-data

# Yeni container başlat
docker-compose up -d weaviate

# Logları izle
docker logs -f rag-weaviate
```

---

## 📞 Destek

Sorun yaşarsanız:
1. Logları kontrol edin
2. Yedek dosyalarının varlığını doğrulayın
3. Docker container'larının durumunu kontrol edin

### Yardımcı Komutlar

```bash
# Tüm container'ları listele
docker ps -a

# Weaviate loglarını izle
docker logs -f rag-weaviate

# Volume'ları listele
docker volume ls

# Docker compose loglarını izle
docker-compose logs -f
```

---

## ✅ Özet

| İşlem | Script | Dosya |
|--------|--------|--------|
| Yedekleme | `./backup-weaviate.sh` | `backups/weaviate-*.tar.gz` |
| Kurtarma | `./restore-weaviate.sh <dosya>` | `backups/weaviate-*.tar.gz` |
| Sürüm Güncelleme | `./upgrade-weaviate.sh <sürüm>` | `docker-compose.yml` |

**Önemli:**
- ✅ Her işlem öncesi yedek alınır
- ✅ Otomatik temizleme (son 7 gün)
- ✅ Container durumu kontrolü
- ✅ Detaylı log çıktısı
- ⚠️  Büyük sürüm atlamalarında dikkatli olun
