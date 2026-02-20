# Database Restore Guide

Bu rehber, backup dosyalarını Coolify ortamına yükleyip restore etmek için adım adım talimatlar içerir.

## Adım 1: Backup Dosyalarını Hazırlayın

Localhost'tan aldığınız backup dosyalarını hazırlayın:
- PostgreSQL: `postgres-pgdump.sql.zip` veya `.sql` dosyası
- Weaviate: `weaviate-backup.json.zip` veya `.json` dosyası

## Adım 2: Restore Sayfasına Gidin

Tarayıcınızda restore sayfasını açın:
```
https://akillirehber.kodleon.com/restore
```

Bu sayfa login gerektirmez ve herkes erişebilir.

## Adım 3: Dosyaları Yükleyin

1. PostgreSQL backup dosyasını seçin
2. Weaviate backup dosyasını seçin (opsiyonel)
3. "Dosyaları Yükle" butonuna tıklayın
4. Upload tamamlandığında dosya yollarını not edin

Dosyalar backend container'da `/app/backups/uploads/` dizinine kaydedilir.

## Adım 4: Coolify Terminaline Girin

Coolify dashboard'da:
1. Backend service'i seçin
2. "Terminal" sekmesine gidin
3. Container shell'e bağlanın

## Adım 5: PostgreSQL Restore

### ZIP dosyası için:
```bash
cd /app/backups/uploads
unzip postgres_*.zip
psql -U raguser -d ragchatbot -f *.sql
```

### Direkt SQL dosyası için:
```bash
psql -U raguser -d ragchatbot -f /app/backups/uploads/postgres_*.sql
```

### Restore durumunu kontrol edin:
```bash
psql -U raguser -d ragchatbot -c "SELECT COUNT(*) FROM users;"
```

## Adım 6: Weaviate Restore (Opsiyonel)

Weaviate restore için Python script kullanmanız gerekir:

```bash
cd /app
python3 << 'EOF'
import json
import zipfile
from pathlib import Path
from app.services.weaviate_service import WeaviateService

# Dosyayı yükle
backup_file = list(Path('/app/backups/uploads').glob('weaviate_*.zip'))[0]

# ZIP'ten çıkar
with zipfile.ZipFile(backup_file, 'r') as zip_ref:
    json_file = [f for f in zip_ref.namelist() if f.endswith('.json')][0]
    with zip_ref.open(json_file) as f:
        backup_data = json.load(f)

# Restore et
weaviate_service = WeaviateService()
total = 0
for collection_name, collection_data in backup_data.get('collections', {}).items():
    course_id = collection_data.get('course_id')
    objects = collection_data.get('objects', [])
    if course_id and objects:
        imported = weaviate_service.import_collection(course_id, objects)
        total += imported
        print(f"Course {course_id}: {imported} objects imported")

print(f"Total: {total} objects imported")
EOF
```

## Adım 7: Giriş Yapın

Restore tamamlandıktan sonra:
```
https://akillirehber.kodleon.com/login
```

Kullanıcı bilgilerinizle giriş yapabilirsiniz.

## Sorun Giderme

### "Permission denied" hatası
```bash
chmod 755 /app/backups/uploads
```

### PostgreSQL bağlantı hatası
```bash
# PostgreSQL'in çalıştığını kontrol edin
pg_isready -U raguser -d ragchatbot
```

### Weaviate bağlantı hatası
```bash
# Weaviate'in çalıştığını kontrol edin
curl http://weaviate:8080/v1/.well-known/ready
```

### Dosya bulunamadı
```bash
# Upload edilen dosyaları listeleyin
ls -lh /app/backups/uploads/
```

## Güvenlik Notu

Restore sayfası herkese açıktır. İlk kurulumdan sonra bu sayfayı devre dışı bırakmak isteyebilirsiniz.

Devre dışı bırakmak için `backend/app/main.py` dosyasından restore router'ı kaldırın:
```python
# app.include_router(restore.router)  # Commented out
```
