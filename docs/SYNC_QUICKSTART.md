# 🚀 Hızlı Başlangıç: Localhost → Coolify Veri Aktarımı

## ⚡ En Hızlı Yöntem (5 Dakika)

### Linux/Mac

```bash
# 1. Script'i çalıştırılabilir yapın
chmod +x sync-to-coolify.sh

# 2. Verileri aktarın
./sync-to-coolify.sh user@your-coolify-server.com

# 3. Coolify'da restore edin
ssh user@your-coolify-server.com
cd ~/rag-backups
docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-*.sql
docker run --rm -v $(docker volume ls -qf name=weaviate):/data -v ~/rag-backups:/backup alpine sh -c 'cd / && tar xzf /backup/weaviate-*.tar.gz'
docker compose restart
```

### Windows

```powershell
# 1. PowerShell'i yönetici olarak açın

# 2. Verileri aktarın
.\sync-to-coolify.ps1 -CoolifyHost "user@your-coolify-server.com"

# 3. Coolify'da restore edin (SSH ile bağlanın)
```

---

## 📋 Ön Gereksinimler

### Localhost'ta

- ✅ Docker çalışıyor olmalı
- ✅ PostgreSQL container'ı çalışıyor: `rag-postgres`
- ✅ Weaviate container'ı çalışıyor: `rag-weaviate`
- ✅ SSH client kurulu (Linux/Mac: varsayılan, Windows: OpenSSH)

### Coolify Sunucusunda

- ✅ SSH erişimi var
- ✅ Docker kurulu
- ✅ Yeterli disk alanı (en az 5GB önerilir)

---

## 🔑 SSH Kurulumu (İlk Kez)

### Linux/Mac

```bash
# SSH key oluşturun (yoksa)
ssh-keygen -t ed25519 -C "your-email@example.com"

# Public key'i sunucuya kopyalayın
ssh-copy-id user@your-coolify-server.com

# Test edin
ssh user@your-coolify-server.com "echo 'Bağlantı başarılı!'"
```

### Windows

```powershell
# OpenSSH'ı etkinleştirin (Windows 10/11)
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# SSH key oluşturun
ssh-keygen -t ed25519 -C "your-email@example.com"

# Public key'i sunucuya kopyalayın
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh user@your-coolify-server.com "cat >> ~/.ssh/authorized_keys"

# Test edin
ssh user@your-coolify-server.com "echo 'Bağlantı başarılı!'"
```

---

## 📦 Ne Aktarılıyor?

| Veri | Boyut (Ortalama) | Süre |
|------|------------------|------|
| PostgreSQL (kullanıcılar, kurslar, ayarlar) | 10-50 MB | 1-2 dk |
| Weaviate (vektör embeddings, dökümanlar) | 100-500 MB | 3-5 dk |
| **Toplam** | **110-550 MB** | **4-7 dk** |

---

## 🎯 Adım Adım Rehber

### 1️⃣ Localhost'ta Yedek Alın

```bash
# Otomatik script ile
./sync-to-coolify.sh user@coolify-server.com

# VEYA manuel olarak
mkdir -p backups
docker exec rag-postgres pg_dump -U raguser -d ragchatbot > backups/postgres.sql
docker run --rm -v rag-weaviate-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/weaviate.tar.gz /data
```

### 2️⃣ Coolify'a Aktarın

```bash
# Otomatik (script zaten yapar)
# VEYA manuel
scp backups/postgres.sql user@coolify-server:~/
scp backups/weaviate.tar.gz user@coolify-server:~/
```

### 3️⃣ Coolify'da Restore Edin

```bash
# Sunucuya bağlanın
ssh user@coolify-server.com

# PostgreSQL restore
docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres.sql

# Weaviate restore
docker run --rm \
    -v $(docker volume ls -qf name=weaviate):/data \
    -v ~/:/backup \
    alpine sh -c 'cd / && tar xzf /backup/weaviate.tar.gz'

# Container'ları yeniden başlatın
cd /path/to/your/project
docker compose restart
```

### 4️⃣ Test Edin

```bash
# Backend health check
curl https://your-api-domain.com/health

# Frontend'e gidin
# https://your-domain.com

# Giriş yapın ve verileri kontrol edin
```

---

## 🔄 Otomatik Senkronizasyon

### Günlük Otomatik Yedekleme

```bash
# Kurulum scripti ile
chmod +x setup-auto-sync.sh
./setup-auto-sync.sh

# Seçenek 1: Localhost (yedek alıcı)
# Seçenek 2: Coolify (yedek alıcı)
# Seçenek 3: İki yönlü senkronizasyon
```

### Manuel Cron Job

```bash
# Crontab'ı düzenleyin
crontab -e

# Her gece saat 2'de yedek al
0 2 * * * /path/to/sync-to-coolify.sh user@coolify-server.com
```

---

## ⚠️ Önemli Notlar

### Güvenlik

- 🔒 SSH key kullanın (password authentication değil)
- 🔒 Yedek dosyalarını şifreleyin (hassas veri içerir)
- 🔒 `.env` dosyasını güvenli şekilde aktarın

### Performans

- ⚡ İlk aktarım uzun sürebilir (veri boyutuna bağlı)
- ⚡ Sonraki aktarımlar daha hızlı (incremental backup)
- ⚡ Sıkıştırma kullanın (gzip/tar.gz)

### Veri Bütünlüğü

- ✅ Aktarım öncesi container'ları durdurun (opsiyonel ama önerilir)
- ✅ Restore sonrası container'ları yeniden başlatın
- ✅ Veritabanı bağlantılarını test edin

---

## 🐛 Sorun Giderme

### "SSH connection refused"

```bash
# SSH servisini kontrol edin
ssh -v user@coolify-server.com

# Port'u kontrol edin (varsayılan: 22)
ssh -p 22 user@coolify-server.com
```

### "Permission denied"

```bash
# SSH key'i ekleyin
ssh-copy-id user@coolify-server.com

# VEYA manuel olarak
cat ~/.ssh/id_ed25519.pub | ssh user@coolify-server.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### "No space left on device"

```bash
# Disk alanını kontrol edin
df -h

# Docker temizliği
docker system prune -a --volumes

# Eski yedekleri silin
find backups -name "*.tar.gz" -mtime +7 -delete
```

### "Container not found"

```bash
# Container isimlerini kontrol edin
docker ps -a

# Volume isimlerini kontrol edin
docker volume ls

# Doğru isimleri kullanın
docker ps --filter "name=postgres"
docker volume ls --filter "name=weaviate"
```

---

## 📞 Yardım

Daha fazla bilgi için:

- 📖 [Detaylı Rehber](./COOLIFY_SYNC_GUIDE.md)
- 🔧 [Sorun Giderme](./COOLIFY_SYNC_GUIDE.md#sorun-giderme)
- 💡 [Gelişmiş Seçenekler](./COOLIFY_SYNC_GUIDE.md#gelişmiş-seçenekler)

---

## ✅ Başarı Kontrol Listesi

- [ ] SSH erişimi çalışıyor
- [ ] Localhost'ta yedek alındı
- [ ] Yedekler Coolify'a aktarıldı
- [ ] PostgreSQL restore edildi
- [ ] Weaviate restore edildi
- [ ] Container'lar yeniden başlatıldı
- [ ] Backend health check başarılı
- [ ] Frontend'e giriş yapıldı
- [ ] Veriler görünüyor
- [ ] Otomatik yedekleme kuruldu (opsiyonel)

---

## 🎉 Tamamlandı!

Verileriniz başarıyla Coolify'a aktarıldı. Artık production ortamında çalışabilirsiniz!

**Sonraki adımlar:**
1. Environment variables'ı Coolify'da ayarlayın
2. Domain ve SSL sertifikası yapılandırın
3. Monitoring ve logging kurun
4. Otomatik yedekleme sistemini aktif edin
