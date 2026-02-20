# Auto-Restore Quickstart Guide

Otomatik yedekleme geri yükleme sistemi, GitHub'a commit edilen yedekleri deployment sırasında otomatik olarak geri yükler.

## Hızlı Başlangıç

### 1. Localhost'ta Yedek Oluştur

```bash
# Tarayıcıda backup sayfasını aç
http://localhost:3000/dashboard/backup

# Admin kullanıcısı olarak giriş yap
# "Create PostgreSQL Backup" butonuna tıkla
# "Create Weaviate Backup" butonuna tıkla
# Her iki yedek dosyasını da indir
```

### 2. Yedekleri Projeye Kopyala

```bash
# İndirilen yedekleri backups klasörüne kopyala
cp ~/Downloads/postgres-*.sql ./backups/
cp ~/Downloads/weaviate-*.json ./backups/

# Veya Windows'ta:
copy %USERPROFILE%\Downloads\postgres-*.sql .\backups\
copy %USERPROFILE%\Downloads\weaviate-*.json .\backups\
```

### 3. GitHub'a Push Et

```bash
git add backups/
git commit -m "Add database backups for auto-restore"
git push origin main
```

### 4. Coolify'da Deploy Et

Coolify otomatik olarak yeni kodu çekecek ve deploy edecek. İlk deployment'ta:

1. ✅ Backend container başlar
2. ✅ Entrypoint script çalışır
3. ✅ Database boş mu kontrol eder
4. ✅ `backups/` klasöründe yedek bulur
5. ✅ PostgreSQL ve Weaviate'i otomatik geri yükler
6. ✅ Uygulama verilerle birlikte başlar

## Nasıl Çalışır?

### Auto-Restore Mantığı

`backend/docker-entrypoint.sh` dosyası şu adımları takip eder:

```bash
1. PostgreSQL ve Weaviate'in hazır olmasını bekle
2. Database migration'ları çalıştır
3. /app/backups klasörünü kontrol et
4. Eğer yedek dosyaları varsa:
   a. Database'in boş olup olmadığını kontrol et (users tablosunda 0 kayıt)
   b. Eğer boşsa:
      - En son postgres-*.sql dosyasını bul ve geri yükle
      - En son weaviate-*.json dosyasını bul ve geri yükle
   c. Eğer boş değilse:
      - Auto-restore'u atla (mevcut veriyi korumak için)
5. Uygulamayı başlat
```

### Dosya İsimlendirme

Yedek dosyaları şu formatta olmalı:

- **PostgreSQL**: `postgres-YYYYMMDD-HHMMSS.sql`
  - Örnek: `postgres-20260205-143000.sql`

- **Weaviate**: `weaviate-YYYYMMDD-HHMMSS.json`
  - Örnek: `weaviate-20260205-143000.json`

Script otomatik olarak **en yeni** yedekleri seçer (dosya değişiklik tarihine göre).

## Logları Kontrol Et

Deployment sonrası auto-restore'un çalışıp çalışmadığını kontrol et:

```bash
# Coolify'da veya SSH ile:
docker logs rag-backend

# Şu mesajları ara:
# "=========================================="
# "Backups found in /app/backups"
# "Database is empty - attempting auto-restore..."
# "Restoring PostgreSQL from: postgres-20260205-143000.sql"
# "✅ PostgreSQL restored successfully"
# "Restoring Weaviate from: weaviate-20260205-143000.json"
# "✅ Weaviate restored successfully (X objects)"
# "Auto-restore completed!"
# "=========================================="
```

## Önemli Notlar

### ✅ Ne Zaman Çalışır?

- **İlk deployment**: Database tamamen boş olduğunda
- **Fresh environment**: Yeni bir Coolify instance'ında
- **Database reset sonrası**: Database'i sıfırladıktan sonra

### ❌ Ne Zaman Çalışmaz?

- **Mevcut veri varsa**: Database'de en az 1 kullanıcı varsa
- **Sonraki deployment'lar**: Uygulama zaten çalışıyorsa
- **Yedek dosyası yoksa**: `backups/` klasörü boşsa

### 🔒 Güvenlik

- Yedek dosyaları hassas veri içerebilir (kullanıcı bilgileri, kurs içeriği)
- Public repository kullanıyorsan yedekleri şifrele
- Production için private repository kullan
- Eski yedekleri düzenli olarak temizle

## Manuel Geri Yükleme

Eğer database boş değilse ve yine de geri yüklemek istiyorsan:

### Yöntem 1: Web Panel

```bash
1. http://your-domain.com/dashboard/backup adresine git
2. Admin olarak giriş yap
3. Yedek dosyalarını upload et
4. "Restore" butonuna tıkla
```

### Yöntem 2: Database'i Sıfırla

```bash
# Coolify'da veya SSH ile:
docker exec -it rag-postgres psql -U raguser -d ragchatbot -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
docker restart rag-backend

# Backend restart olduğunda auto-restore çalışacak
```

## Sorun Giderme

### Problem: Auto-restore çalışmadı

**Çözüm 1**: Database'in boş olup olmadığını kontrol et

```bash
docker exec -it rag-backend python -c "
from app.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
result = db.execute(text('SELECT COUNT(*) FROM users')).scalar()
print(f'Users: {result}')
db.close()
"
```

**Çözüm 2**: Yedek dosyalarının container'da olup olmadığını kontrol et

```bash
docker exec -it rag-backend ls -la /app/backups
```

### Problem: Yedek dosyaları bulunamadı

**Çözüm**: Volume mount'u kontrol et

```bash
# docker-compose.coolify.yml dosyasında şu satır olmalı:
volumes:
  - ./backups:/app/backups:ro
```

### Problem: Restore başarısız oldu

**Çözüm**: Detaylı hata mesajları için logları kontrol et

```bash
docker logs rag-backend 2>&1 | grep -A 20 "auto-restore"
```

## Örnek Senaryo

### Senaryo: Localhost'tan Coolify'a Tam Veri Transferi

```bash
# 1. Localhost'ta yedek oluştur
curl -X POST http://localhost:8000/api/admin/backup/create/postgres \
  -H "Authorization: Bearer YOUR_TOKEN"

curl -X POST http://localhost:8000/api/admin/backup/create/weaviate \
  -H "Authorization: Bearer YOUR_TOKEN"

# 2. Yedekleri indir (web panelden)
# http://localhost:3000/dashboard/backup

# 3. Yedekleri projeye kopyala
cp ~/Downloads/postgres-20260205-143000.sql ./backups/
cp ~/Downloads/weaviate-20260205-143000.json ./backups/

# 4. Git'e commit et
git add backups/
git commit -m "Add production backups"
git push origin main

# 5. Coolify'da deploy et
# Coolify otomatik olarak yeni kodu çekecek ve deploy edecek

# 6. Logları kontrol et
docker logs rag-backend | grep "auto-restore"

# 7. Uygulamayı test et
curl http://your-domain.com/api/health
```

## Volume Mapping

### docker-compose.yml (Localhost)

```yaml
backend:
  volumes:
    - ./backups:/app/backups:ro
```

### docker-compose.coolify.yml (Production)

```yaml
backend:
  volumes:
    - ./backups:/app/backups:ro
```

Her iki ortamda da aynı volume mapping kullanılıyor, böylece yedekler her yerde aynı şekilde çalışıyor.

## Sonuç

Auto-restore sistemi sayesinde:

- ✅ Localhost'tan production'a veri transferi kolay
- ✅ Yeni environment'lar hızlıca kurulabilir
- ✅ Disaster recovery basitleşir
- ✅ Test environment'ları production verisiyle doldurulabilir
- ✅ Manual restore adımları ortadan kalkar

Herhangi bir sorun yaşarsan `backups/README.md` dosyasına bak veya logları kontrol et.
