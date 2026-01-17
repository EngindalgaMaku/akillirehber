# Coolify ile Hetzner'e Deployment Rehberi

Bu rehber, RAG Educational Chatbot sistemini Coolify kullanarak Hetzner sunucusuna deploy etmeyi açıklar.

## Gereksinimler

- Hetzner Cloud hesabı
- Domain adı (opsiyonel ama önerilir)
- GitHub/GitLab hesabı (repo için)

## Adım 1: Hetzner Sunucu Oluşturma

1. [Hetzner Cloud Console](https://console.hetzner.cloud)'a giriş yapın
2. Yeni proje oluşturun veya mevcut projeyi seçin
3. "Add Server" butonuna tıklayın
4. Ayarlar:
   - **Location**: Nuremberg veya Helsinki (Türkiye'ye yakın)
   - **Image**: Ubuntu 24.04
   - **Type**: CX21 (4GB RAM) veya CX31 (8GB RAM) - önerilen
   - **SSH Key**: Mevcut key'inizi ekleyin veya yeni oluşturun
5. Sunucuyu oluşturun ve IP adresini not edin

## Adım 2: Coolify Kurulumu

SSH ile sunucuya bağlanın:

```bash
ssh root@SUNUCU_IP
```

Coolify'ı kurun:

```bash
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
```

Kurulum tamamlandıktan sonra:
- `http://SUNUCU_IP:8000` adresine gidin
- Admin hesabı oluşturun
- İlk kurulum sihirbazını tamamlayın

## Adım 3: GitHub Repository Bağlantısı

1. Coolify dashboard'da "Sources" > "Add" > "GitHub App" seçin
2. GitHub hesabınızı bağlayın
3. Repository'ye erişim izni verin

## Adım 4: Proje Deploy Etme

### Yöntem A: Docker Compose ile (Önerilen)

1. Coolify'da "Projects" > "Add" ile yeni proje oluşturun
2. "Add Resource" > "Docker Compose" seçin
3. GitHub repository'nizi seçin
4. Docker Compose dosyası olarak `docker-compose.coolify.yml` seçin
5. Environment Variables ekleyin (aşağıya bakın)
6. "Deploy" butonuna tıklayın

### Yöntem B: Ayrı Servisler

Her servisi ayrı ayrı deploy edebilirsiniz:

1. PostgreSQL: "Add Resource" > "Database" > "PostgreSQL"
2. Weaviate: "Add Resource" > "Docker Image" > `semitechnologies/weaviate:1.27.0`
3. Backend: "Add Resource" > "Application" > Dockerfile
4. Frontend: "Add Resource" > "Application" > Dockerfile
5. RAGAS: "Add Resource" > "Application" > Dockerfile

## Adım 5: Environment Variables

Coolify'da şu environment variable'ları ayarlayın:

### Zorunlu

```env
# Database
POSTGRES_USER=raguser
POSTGRES_PASSWORD=<güçlü-şifre-oluşturun>
POSTGRES_DB=ragchatbot

# Security
SECRET_KEY=<64-karakter-rastgele-string>

# API URL (domain'inize göre değiştirin)
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

### LLM API Keys (en az biri gerekli)

```env
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
COHERE_API_KEY=...
DASHSCOPE_API_KEY=...
```

### Opsiyonel

```env
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080
ENVIRONMENT=production
RAGAS_MODEL=openai/gpt-4o-mini
```

## Adım 6: Domain Ayarları

### Coolify'da Domain Ekleme

1. Frontend servisine gidin > "Settings" > "Domains"
2. Domain ekleyin: `yourdomain.com`
3. Backend için: `api.yourdomain.com`

### DNS Ayarları

Domain sağlayıcınızda A kayıtları ekleyin:

```
yourdomain.com      A    SUNUCU_IP
api.yourdomain.com  A    SUNUCU_IP
```

### SSL Sertifikası

Coolify otomatik olarak Let's Encrypt SSL sertifikası oluşturur.

## Adım 7: Reverse Proxy Ayarları

Coolify'ın Traefik proxy'si otomatik yapılandırılır. Manuel ayar gerekirse:

Frontend için:
- Port: 3000
- Path: /

Backend için:
- Port: 8000
- Path: /api

## Adım 8: İlk Kullanıcı Oluşturma

Deploy tamamlandıktan sonra:

1. `https://yourdomain.com/register` adresine gidin
2. İlk öğretmen hesabını oluşturun
3. Giriş yapın ve sistemi kullanmaya başlayın

## Sorun Giderme

### Container Logları

Coolify dashboard'dan her servisin loglarını görüntüleyebilirsiniz.

### Database Bağlantı Hatası

```bash
# Sunucuda container'ları kontrol edin
docker ps
docker logs rag-postgres
```

### Weaviate Başlamıyor

Weaviate'in başlaması 30-60 saniye sürebilir. Healthcheck'leri bekleyin.

### Frontend API Bağlantı Hatası

1. `NEXT_PUBLIC_API_URL` doğru ayarlandığından emin olun
2. Backend'in çalıştığını kontrol edin
3. CORS ayarlarını kontrol edin

## Yedekleme

### Database Yedekleme

```bash
# Coolify'da scheduled backup ayarlayabilirsiniz
# veya manuel:
docker exec rag-postgres pg_dump -U raguser ragchatbot > backup.sql
```

### Volume Yedekleme

```bash
# Weaviate data
docker run --rm -v rag-weaviate-data:/data -v $(pwd):/backup alpine tar czf /backup/weaviate-backup.tar.gz /data

# PostgreSQL data
docker run --rm -v rag-postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

## Güncelleme

1. GitHub'a yeni commit push edin
2. Coolify'da "Redeploy" butonuna tıklayın
3. Veya webhook ile otomatik deploy ayarlayın

## Kaynak Kullanımı

Önerilen minimum sunucu özellikleri:
- **RAM**: 4GB (8GB önerilir)
- **CPU**: 2 vCPU
- **Disk**: 40GB SSD

Servis bazlı kaynak kullanımı:
- PostgreSQL: ~200MB RAM
- Weaviate: ~500MB-1GB RAM
- Backend: ~300MB RAM
- Frontend: ~200MB RAM
- RAGAS: ~500MB RAM
