# HiveFlow — Meal Planning with AI Support

**Bitirme projesi AI modülü** · HiveFlow Employee Management System / Management Panel

Bu depo, HiveFlow yönetim paneli için geliştirilen **Meal Planning with AI Support** modülünün bağımsız yapay zeka katmanını içerir. Panel (PHP · MySQL · Bootstrap 5 · cPanel) artık canlıda kullanılmıyor; **AI öneri motoru ve Flask API’leri çalışır durumda** tutulmuştur.

---

## Proje özeti

Kullanıcı beslenme geçmişi ve öğün özellikleri (kalori, protein, diyet kısıtlamaları) **Multi-input Neural Network** mimarisinde işlenerek **Binary Classification** tabanlı kişiselleştirilmiş bir tavsiye motoru geliştirildi. Tahmin modeli **Flask-CORS** ile RESTful API’ye dönüştürüldü; canlı entegrasyon döneminde **Ngrok** üzerinden PHP/MySQL backend’e bağlandı. UI tarafında HTML, CSS, JavaScript ve Bootstrap 5 kullanıldı; model eğitimi ve API yayını Google Colab üzerinden de çalıştırılabilir.

| Katman | Teknoloji | Rol |
|--------|-----------|-----|
| Veri işleme | Python, Pandas, NumPy | Öğün kataloğu, geçmiş, normalizasyon |
| NLP benzerlik | Scikit-learn (TF-IDF, cosine) | Metin tercihine göre hızlı öneri |
| Deep Learning | TensorFlow / Keras | Multi-input NN + binary classification |
| API | Flask, Flask-CORS | REST endpoint’ler |
| Tünel (opsiyonel) | Ngrok / pyngrok | PHP backend entegrasyonu |
| Panel (arşiv) | PHP, MySQL, Bootstrap 5, cPanel | Yönetim UI — artık kapalı |

---

## Mimari

```
[ Kullanıcı tercihi / geçmiş ]
            │
            ▼
┌───────────────────────────┐     ┌────────────────────────────┐
│  NLP Similarity API       │     │  Deep Learning API         │
│  TF-IDF + protein boost   │     │  Multi-input Keras NN      │
│  POST /recommendations    │     │  POST /smart-recommendations│
└─────────────┬─────────────┘     └──────────────┬─────────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                    Flask-CORS REST JSON
                             │
              (opsiyonel) Ngrok public URL
                             │
                    PHP / MySQL panel (arşiv)
```

Detaylı akış: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Depo yapısı

```
├── api/
│   ├── deep_learning_recommender.py   # Multi-input NN + /smart-recommendations
│   └── nlp_similarity_recommender.py  # TF-IDF benzerlik + /recommendations
├── notebooks/
│   ├── 01_deep_learning_api.ipynb     # Colab’da eğitim + API (DL)
│   └── 02_nlp_similarity_api.ipynb    # Colab’da TF-IDF API
├── docs/
│   └── ARCHITECTURE.md
├── requirements.txt
├── .env.example
└── README.md
```

Eski Colab dosya adları (`6.ipynb`, `ml2.ipynb`) anlamlı isimlere taşındı.

---

## İki öneri motoru

### 1) Deep Learning — Multi-input Neural Network

Dosya: [`api/deep_learning_recommender.py`](api/deep_learning_recommender.py)

- **Girdiler:** kullanıcı özellikleri (ortalama kalori/protein + tag tercihi), öğün nutrisyonu, kategori one-hot, tag vektörü
- **Çıktı:** sigmoid skoru (0–1) → öğünün kullanıcıya uygun olma olasılığı
- **Eğitim:** API ayağa kalkarken pozitif (geçmişte tüketilen) / negatif örneklerle binary cross-entropy
- **Filtreler:** `calorie_range`, `protein_range`, `allowed_categories` (ör. vegan / vegetarian)

```http
GET  /
POST /smart-recommendations
```

Örnek yanıt:

```json
{
  "success": true,
  "recommendations": [
    {
      "meal_id": 6,
      "name": "Vegan Buddha Bowl",
      "category": "vegan",
      "calories": 450,
      "protein": 15,
      "score": 0.91
    }
  ]
}
```

### 2) NLP Similarity — TF-IDF

Dosya: [`api/nlp_similarity_recommender.py`](api/nlp_similarity_recommender.py)

- Kullanıcı metin tercihini (`"chicken spicy"` vb.) TF-IDF ile vektörleştirir
- Cosine similarity + protein / exact-match boosting
- Derin öğrenmeye göre daha hafif; tercih metnine hızlı yanıt

```http
GET  /
POST /recommendations
Content-Type: application/json

{ "preference": "tofu vegan spicy", "count": 5 }
```

---

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # veya: cp .env.example .env
```

`.env` içinde isteğe bağlı:

```env
NGROK_AUTH_TOKEN=your_ngrok_token_here
PORT=5000
```

> Ngrok token **zorunlu değildir**. Token yoksa API yalnızca localhost’ta çalışır.

---

## Çalıştırma

### Deep Learning API

```bash
python api/deep_learning_recommender.py
```

İlk açılışta model eğitilir; ardından:

- Yerel: `http://127.0.0.1:<PORT>/smart-recommendations`
- Ngrok varsa konsolda public URL basılır

### NLP Similarity API

```bash
python api/nlp_similarity_recommender.py
```

- Yerel: `http://127.0.0.1:<PORT>/recommendations`

### Google Colab

1. `notebooks/01_deep_learning_api.ipynb` veya `02_nlp_similarity_api.ipynb` dosyasını Colab’a yükleyin
2. Gerekirse Colab secrets / env ile `NGROK_AUTH_TOKEN` tanımlayın
3. Hücreyi çalıştırın — eğitim + Flask + (opsiyonel) Ngrok tüneli

---

## PHP / MySQL entegrasyonu (arşiv notu)

HiveFlow paneli kapalıdır; entegrasyon dönemi şöyleydi:

1. Colab veya sunucuda Flask API + Ngrok
2. PHP tarafında Ngrok URL’sine `POST` (JSON)
3. MySQL’deki kullanıcı / öğün verisi paneli besler; AI skoru öneri listesini üretir
4. Bootstrap 5 arayüz önerileri listeler

Bugün aynı API’ler bağımsız olarak test edilebilir; panel kodu bu deponun kapsamında değildir.

---

## Güvenlik notu

Ngrok auth token’ları kod içine gömülmez. Ortam değişkeni veya Colab secret kullanın. Eski notebook’larda hard-coded token vardıysa **yenileyin / revoke edin**.

---

## Lisans / akademik kullanım

Bitirme projesi kapsamında geliştirilmiştir. Akademik alıntı ve portföy sunumu için serbestçe kullanılabilir.
