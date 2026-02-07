# Coolify Veri Senkronizasyon Rehberi

Bu rehber, localhost'taki RAG Educational Chatbot verilerinizi Coolify ortamına aktarmanız için hazırlanmıştır.

## 📋 İçindekiler

1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Manuel Yöntem](#manuel-yöntem)
3. [Otomatik Senkronizasyon](#otomatik-senkronizasyon)
4. [Sürekli Senkronizasyon](#sürekli-senkronizasyon)
5. [Sorun Giderme](#sorun-giderme)

---

## 🚀 Hızlı Başlangıç

### Ön Gereksinimler

```bash
# SSH erişimi test edin
ssh user@your-coolify-server.com

# SSH key yoksa ekleyin
ssh-copy-id user@your-coolify-server.com
```

### Tek Komutla Aktarım

```bash
# Script'i çalıştırılabilir yapın
chmod +x sync-to-coolify.sh

# Verileri aktarın
./sync-to-coolify.sh user@your-coolify-server.com
```

---

## 📦 Manuel Yöntem

### 1. PostgreSQL Veritabanı Aktarımı

#### Localhost'ta Yedek Alın

```bash
# Yedek dizini oluşturun
mkdir -p backups

# PostgreSQL dump alın
docker exec rag-postgres pg_dump -U raguser -d ragchatbot > backups/postgres-backup.sql

# Yedeği sıkıştırın (opsiyonel)
gzip backups/postgres-backup.sql
```

#### Coolify'a Aktarın

```bash
# Yedeği sunucuya kopyalayın
scp backups/postgres-backup.sql user@coolify-server:~/

# Sunucuya bağlanın
ssh user@coolify-server

# Restore edin
docker exec -i $(docker ps -qf name=postgres) psql -U raguser -d ragchatbot < postgres-backup.sql
```

### 2. Weaviate Vektör Veritabanı Aktarımı

#### Localhost'ta Yedek Alın

```bash
# Weaviate volume'unu yedekleyin
docker run --rm \
    -v rag-weaviate-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/weaviate-backup.tar.gz /data
```

#### Coolify'a Aktarın

```bash
# Yedeği sunucuya kopyalayın
scp backups/weaviate-backup.tar.gz user@coolify-server:~/

# Sunucuya bağlanın
ssh user@coolify-server

# Weaviate volume adını bulun
docker volume ls | grep weaviate

# Restore edin
docker run --rm \
    -v <weaviate-volume-name>:/data \
    -v ~/:/backup \
    alpine sh -c 'cd / && tar xzf /backup/weaviate-backup.tar.gz'

# Container'ı yeniden başlatın
docker compose restart weaviate
```

### 3. Environment Variables Aktarımı

```bash
# .env dosyasını kopyalayın (hassas bilgiler içerir!)
scp .env user@coolify-server:~/project-path/

# VEYA Coolify UI'dan manuel olarak ekleyin
```

---

## 🔄 Otomatik Senkronizasyon

### Günlük Otomatik Yedekleme

Coolify sunucusunda cron job oluşturun:

```bash
# Crontab'ı düzenleyin
crontab -e

# Her gece saat 2'de yedek alın
0 2 * * * /path/to/backup-script.sh
```

### Backup Script Örneği

```bash
#!/bin/bash
# backup-daily.sh

BACKUP_DIR="/backups/rag-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL
docker exec rag-postgres pg_dump -U raguser -d ragchatbot > $BACKUP_DIR/postgres.sql

# Weaviate
docker run --rm \
    -v rag-weaviate-data:/data \
    -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/weaviate.tar.gz /data

# Eski yedekleri temizle (30 günden eski)
find /backups -name "rag-*" -mtime +30 -exec rm -rf {} \;
```

---

## 🔁 Sürekli Senkronizasyon (Bi-directional Sync)

### Rsync ile Otomatik Senkronizasyon

```bash
#!/bin/bash
# continuous-sync.sh

COOLIFY_HOST="user@coolify-server"
LOCAL_BACKUP="./backups"
REMOTE_BACKUP="~/rag-backups"

# Her 5 dakikada bir senkronize et
while true; do
    echo "Senkronizasyon başlatılıyor..."
    
    # Localhost'tan Coolify'a
    rsync -avz --progress $LOCAL_BACKUP/ $COOLIFY_HOST:$REMOTE_BACKUP/
    
    echo "Senkronizasyon tamamlandı. 5 dakika bekleniyor..."
    sleep 300
done
```

### Systemd Service Olarak Çalıştırma

```bash
# /etc/systemd/system/rag-sync.service
[Unit]
Description=RAG Continuous Sync Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/path/to/continuous-sync.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Servisi etkinleştirin
sudo systemctl enable rag-sync
sudo systemctl start rag-sync
sudo systemctl status rag-sync
```

---

## 🔧 Gelişmiş Seçenekler

### 1. Incremental Backup (Artımlı Yedekleme)

PostgreSQL için WAL (Write-Ahead Logging) kullanın:

```bash
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'rsync -a %p user@coolify-server:/wal-archive/%f'
```

### 2. Weaviate Backup API Kullanımı

```bash
# Weaviate'in kendi backup API'sini kullanın
curl -X POST http://localhost:8080/v1/backups/filesystem \
  -H "Content-Type: application/json" \
  -d '{
    "id": "backup-'$(date +%Y%m%d)'",
    "include": ["*"]
  }'
```

### 3. Docker Volume Replication

```bash
# Volume'ları doğrudan kopyalayın
docker run --rm \
    -v rag-weaviate-data:/from \
    -v new-weaviate-data:/to \
    alpine sh -c "cd /from && cp -av . /to"
```

---

## 🐛 Sorun Giderme

### SSH Bağlantı Sorunları

```bash
# SSH key'i test edin
ssh -v user@coolify-server

# Key yoksa oluşturun
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@coolify-server
```

### PostgreSQL Restore Hataları

```bash
# Veritabanını temizleyin
docker exec -it rag-postgres psql -U raguser -d ragchatbot -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Tekrar restore edin
docker exec -i rag-postgres psql -U raguser -d ragchatbot < backup.sql
```

### Weaviate Volume Bulunamıyor

```bash
# Volume'ları listeleyin
docker volume ls

# Volume'u inspect edin
docker volume inspect rag-weaviate-data

# Yeni volume oluşturun
docker volume create rag-weaviate-data
```

### Disk Alanı Yetersiz

```bash
# Disk kullanımını kontrol edin
df -h

# Docker temizliği yapın
docker system prune -a --volumes

# Eski yedekleri silin
find ./backups -name "*.tar.gz" -mtime +7 -delete
```

---

## 📊 Veri Boyutu Optimizasyonu

### PostgreSQL Vacuum

```bash
# Veritabanını optimize edin
docker exec rag-postgres psql -U raguser -d ragchatbot -c "VACUUM FULL ANALYZE;"
```

### Weaviate Compaction

```bash
# Weaviate'i optimize edin (API üzerinden)
curl -X POST http://localhost:8080/v1/schema/compact
```

---

## 🔐 Güvenlik Önerileri

1. **Yedekleri Şifreleyin**
```bash
# GPG ile şifreleme
gpg --symmetric --cipher-algo AES256 backup.sql
```

2. **SSH Key Kullanın**
```bash
# Password authentication'ı devre dışı bırakın
# /etc/ssh/sshd_config
PasswordAuthentication no
```

3. **Backup Dosyalarını Koruyun**
```bash
# Sadece owner okuyabilsin
chmod 600 backups/*.sql
```

---

## 📝 Checklist

- [ ] SSH erişimi test edildi
- [ ] PostgreSQL yedeği alındı
- [ ] Weaviate yedeği alındı
- [ ] Environment variables kopyalandı
- [ ] Coolify'da restore edildi
- [ ] Container'lar yeniden başlatıldı
- [ ] Uygulama test edildi
- [ ] Otomatik yedekleme kuruldu

---

## 🆘 Yardım

Sorun yaşıyorsanız:

1. Log'ları kontrol edin: `docker compose logs -f`
2. Container durumunu kontrol edin: `docker ps -a`
3. Volume'ları kontrol edin: `docker volume ls`
4. Disk alanını kontrol edin: `df -h`

---

## 📚 Ek Kaynaklar

- [Coolify Documentation](https://coolify.io/docs)
- [PostgreSQL Backup Guide](https://www.postgresql.org/docs/current/backup.html)
- [Weaviate Backup Documentation](https://weaviate.io/developers/weaviate/configuration/backups)
- [Docker Volume Management](https://docs.docker.com/storage/volumes/)
