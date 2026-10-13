# Mimari — HiveFlow Meal Planning AI

## Amaç

Çalışan / kullanıcı için kişiselleştirilmiş öğün önerisi üretmek. Sistem iki tamamlayıcı yaklaşımı birleştirir:

1. **Deep Learning (asıl motor):** geçmiş tüketim + nutrisyon + diyet kısıtı → binary classification skoru  
2. **NLP Similarity (hafif motor):** serbest metin tercih → TF-IDF benzerlik

## Deep Learning veri akışı

```
vegan_meal_plan (günlük kahvaltı/öğle/akşam)
        │
        ▼
user_history_df  (user_id, meal_id, timestamp)
        │
        ├──────────────► calculate_user_features()
        │                  • avg calories_norm, protein_norm
        │                  • tag preference vector
        │
meals_df (50 öğün: category, calories, protein, tags, dietary)
        │
        ├──────────────► prepare_meal_features()
        │                  • nutritional (2)
        │                  • category one-hot
        │                  • multi-hot tags
        │
        ▼
create_recommendation_model()
  Inputs:  user | meal_nutri | category | tags
  Dense + Dropout + Concatenate
  Output: sigmoid (P(recommend))
        │
        ▼
train_model_on_startup()
  positive = history meals
  negative = allowed-category, uneaten meals
  loss = binary_crossentropy
        │
        ▼
get_recommendations_api()
  score all meals → filter by category / calorie / protein → top-N
```

## Model şeması (Keras)

| Input | Shape | İşlem |
|-------|-------|--------|
| `user_input` | `(num_user_features,)` | Dense(64) + Dropout |
| `meal_input` | `(2,)` calories/protein norm | Dense(64) + Dropout |
| `category_input` | `(num_categories,)` | doğrudan concat |
| `tag_input` | `(num_tags,)` | Dense(64) + Dropout |
| Concat → Dense(128) → Dense(64) → Dense(1, sigmoid) | | |

Düzenlileştirme: L2 (`0.001`), Adam `lr=0.0005`, EarlyStopping (val_loss).

## NLP Similarity veri akışı

```
preference text  ("grilled chicken healthy")
        │
        ▼
boost tags by matching main_protein / exact name
        │
        ▼
TfidfVectorizer (1–2 grams) + linear_kernel cosine
        │
        ▼
top-N meals with score / protein / tags
```

## API sözleşmesi

### Deep Learning — `/smart-recommendations`

- Method: `GET` veya `POST`
- Gövde opsiyonel (varsayılan kullanıcı geçmişi ve global preferences kullanılır)
- Yanıt: `{ success, recommendations: [{ meal_id, name, category, calories, protein, tags, score }] }`

### NLP — `/recommendations`

- Method: `POST`
- Body: `{ "preference": string, "count": number }`
- Yanıt: `{ success, query, recommendations: [{ name, score, protein, tags }] }`

## Deployment geçmişi

| Ortam | Rol | Durum |
|-------|-----|--------|
| Google Colab | Model eğitimi + Flask + Ngrok | Çalıştırılabilir (notebooks/) |
| Ngrok | Public tunnel → PHP | Opsiyonel |
| cPanel / PHP / MySQL | HiveFlow Management Panel | **Kapalı (arşiv)** |
| Bu repo (`api/*.py`) | Bağımsız AI modülü | **Aktif** |

## Neden iki motor?

- **DL:** geçmiş ve nutrisyon kısıtlarıyla “bu kullanıcı için uygun mu?” skorlar  
- **NLP:** anlık metin araması / keşif; eğitimsiz, düşük gecikme  

Panel döneminde PHP her iki endpoint’e de Ngrok URL üzerinden istek atabiliyordu.
