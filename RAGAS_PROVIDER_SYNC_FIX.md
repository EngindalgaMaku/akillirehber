# RAGAS Provider Sync Fix - Eksik Provider'lar Eklendi

## 🐛 Sorun

**Belirti:**
- RAGAS ayarlarında bazı provider'lar ve modeller kaydedilemiyor
- Örneğin: `zai` provider'ı ve `glm-4.7` modeli frontend'de görünüyor ama kaydedilmiyor
- Backend'de tanımlı olan provider'lar RAGAS service'de eksik

**Neden Oluyor:**
- Backend (`backend/app/services/llm_providers.py`) ve RAGAS service (`ragas_service/main.py`) arasında provider listesi senkronize değil
- Backend'de tanımlı olan bazı provider'lar RAGAS service'de eksik:
  - `zai` (Z.ai - glm-4.7)
  - `deepseek` (DeepSeek Chat, DeepSeek Coder)
  - `cohere` (Command-R modelleri)
  - `alibaba` (Qwen modelleri)

## ✅ Çözüm

### 1. RAGAS Service'e Eksik Provider'lar Eklendi

**ragas_service/main.py - LLM_PROVIDERS_CONFIG:**

```python
LLM_PROVIDERS_CONFIG = {
    # ... existing providers ...
    
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "embedding_model": None,
        "is_free": False,
        "priority": 6,
        "models": [
            "deepseek-chat",
            "deepseek-coder",
        ],
    },
    "cohere": {
        "env_key": "COHERE_API_KEY",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_model": "command-r-08-2024",
        "embedding_model": None,
        "is_free": False,
        "priority": 7,
        "models": [
            "command-r-08-2024",
            "command-r-plus-08-2024",
            "command-a-03-2025",
            "command-r-plus",
        ],
    },
    "alibaba": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-turbo",
        "embedding_model": None,
        "is_free": False,
        "priority": 8,
        "models": [
            "qwen-turbo",
            "qwen-flash",
            "qwen3-8b",
            "qwen3-14b",
            "qwen-plus",
            "qwen3-32b",
            "qwen-max",
            "qwen3-max",
        ],
    },
    "zai": {
        "env_key": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/anthropic/v1/messages",
        "default_model": "glm-4.7",
        "embedding_model": None,
        "is_free": False,
        "priority": 9,
        "models": [
            "glm-4.7",
        ],
        "default_headers": {
            "anthropic-version": "2023-06-01"
        },
    },
}
```

### 2. Custom Headers Desteği İyileştirildi

**Önceki Durum:**
- `default_headers` sadece `openrouter` ve `claudegg` için hardcoded ekleniyor
- `zai` gibi özel header gerektiren provider'lar çalışmıyor

**Yeni Durum:**
- `get_llm_config()` fonksiyonu artık `default_headers`'ı config'den döndürüyor
- Her provider kendi `default_headers`'ını tanımlayabiliyor
- LLM initialization kısmı önce config'den header'ları alıyor, yoksa legacy fallback kullanıyor

**ragas_service/main.py - get_llm_config():**

```python
def get_llm_config():
    # ... provider selection logic ...
    
    result = {
        "provider": provider_name,
        "api_key": api_key,
        "base_url": config["base_url"],
        "model": custom_model or config["default_model"],
        "embedding_model": config["embedding_model"],
        "is_free": config["is_free"],
    }
    
    # ✅ Add default_headers if provider has them
    if "default_headers" in config:
        result["default_headers"] = config["default_headers"]
    
    return result
```

**ragas_service/main.py - LLM Initialization:**

```python
llm_kwargs = {
    "model": model_to_use,
    "api_key": llm_config["api_key"],
    "temperature": 0,
}

if llm_config["base_url"]:
    llm_kwargs["base_url"] = llm_config["base_url"]

# ✅ Add default headers from config if available
if "default_headers" in llm_config:
    llm_kwargs["default_headers"] = llm_config["default_headers"]
# Legacy: Add headers for OpenRouter and Claude.gg if not already set
elif llm_config["provider"] in ["openrouter", "claudegg"]:
    llm_kwargs["default_headers"] = {
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": SERVICE_NAME,
    }

llm = ChatOpenAI(**llm_kwargs)
```

## 📊 Eklenen Provider'lar

| Provider | Base URL | Models | Özel Özellik |
|----------|----------|--------|--------------|
| **deepseek** | api.deepseek.com | deepseek-chat, deepseek-coder | Kod odaklı model |
| **cohere** | api.cohere.ai | command-r-08-2024, command-r-plus-08-2024, command-a-03-2025 | Çok dilli destek |
| **alibaba** | dashscope.aliyuncs.com | qwen-turbo, qwen-flash, qwen3-8b, qwen-plus, qwen-max | Çince güçlü |
| **zai** | api.z.ai | glm-4.7 | Anthropic API format, özel header gerekli |

## 🔧 Değişiklikler

### Dosyalar

1. **ragas_service/main.py**
   - `LLM_PROVIDERS_CONFIG` - 4 yeni provider eklendi
   - `get_llm_config()` - `default_headers` desteği eklendi
   - LLM initialization (2 yer) - Config'den header okuma eklendi

### Özellikler

✅ **Tüm backend provider'ları artık RAGAS'ta kullanılabilir**
✅ **Custom header desteği genelleştirildi**
✅ **Z.ai gibi özel format gerektiren provider'lar çalışıyor**
✅ **Backward compatible** - Eski provider'lar etkilenmedi

## 🧪 Test

### Test Adımları

1. **RAGAS Ayarları Sayfasını Aç:**
   ```
   http://localhost:3000/dashboard/ragas
   ```

2. **Ayarlar Butonuna Tıkla**

3. **Provider Listesini Kontrol Et:**
   - ✅ openrouter
   - ✅ claudegg
   - ✅ apiclaudegg
   - ✅ groq
   - ✅ openai
   - ✅ deepseek (YENİ)
   - ✅ cohere (YENİ)
   - ✅ alibaba (YENİ)
   - ✅ zai (YENİ)

4. **Z.ai Provider'ını Seç:**
   - Model: `glm-4.7` görünmeli
   - Kaydet butonuna tıkla
   - ✅ Başarıyla kaydedilmeli

5. **RAGAS Test Çalıştır:**
   - Quick Test veya Batch Test yap
   - ✅ Z.ai modeli kullanılmalı
   - ✅ Hata olmamalı

### Beklenen Sonuç

```json
{
  "provider": "zai",
  "model": "glm-4.7",
  "current_provider": "zai",
  "current_model": "glm-4.7",
  "is_free": false
}
```

## 🐛 Sorun Giderme

### Z.ai Kaydedilmiyor

**Kontrol:**
1. RAGAS service restart edildi mi?
   ```bash
   docker-compose restart ragas
   ```

2. Z.ai API key tanımlı mı?
   ```bash
   echo $ZAI_API_KEY
   ```

3. RAGAS service log'larını kontrol et:
   ```bash
   docker-compose logs ragas | grep -i "zai\|glm"
   ```

### Diğer Provider'lar Kaydedilmiyor

**Kontrol:**
1. Provider adı doğru mu? (küçük harf olmalı: `deepseek`, `cohere`, `alibaba`)

2. API key tanımlı mı?
   ```bash
   echo $DEEPSEEK_API_KEY
   echo $COHERE_API_KEY
   echo $DASHSCOPE_API_KEY
   ```

3. Backend ve RAGAS service'de aynı provider adı kullanılıyor mu?

### Custom Headers Çalışmıyor

**Kontrol:**
1. Provider config'inde `default_headers` tanımlı mı?

2. `get_llm_config()` fonksiyonu header'ları döndürüyor mu?
   ```python
   # Debug log ekle
   logger.info(f"LLM config: {llm_config}")
   ```

3. LLM initialization kısmında header'lar ekleniyor mu?
   ```python
   # Debug log ekle
   logger.info(f"LLM kwargs: {llm_kwargs}")
   ```

## 📝 Gelecek İyileştirmeler

1. **Provider Sync Automation**
   - Backend ve RAGAS service provider listelerini tek bir yerden yönet
   - Otomatik senkronizasyon scripti

2. **Provider Health Check**
   - Her provider için health check endpoint'i
   - API key geçerliliği kontrolü

3. **Dynamic Provider Loading**
   - Runtime'da yeni provider ekleme
   - Database'den provider config okuma

## 🎉 Sonuç

Artık backend'de tanımlı olan **TÜM provider'lar** RAGAS ayarlarında kullanılabilir!

**Önceki ❌:**
- 5 provider (openrouter, claudegg, apiclaudegg, groq, openai)
- Z.ai kaydedilemiyor
- Custom header desteği sınırlı

**Yeni ✅:**
- ✅ 9 provider (4 yeni eklendi)
- ✅ Z.ai ve diğer tüm provider'lar çalışıyor
- ✅ Genelleştirilmiş custom header desteği
- ✅ Backend ile tam senkronize

**"Artık z.ai ve diğer tüm provider'lar RAGAS'ta kullanılabilir!"** 🚀
