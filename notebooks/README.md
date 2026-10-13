# Colab notebooks

Bu klasör, projenin Google Colab üzerinden canlıya alındığı dönemdeki API başlatma akışını korur.

| Dosya | Eski ad | Ne yapar? |
|-------|---------|-----------|
| `01_deep_learning_api.ipynb` | `6.ipynb` | Multi-input NN eğitimi + `/smart-recommendations` |
| `02_nlp_similarity_api.ipynb` | `ml2.ipynb` | TF-IDF benzerlik + `/recommendations` |

Yerel geliştirme için `api/` altındaki `.py` dosyalarını tercih edin. Colab’da çalıştırırken `NGROK_AUTH_TOKEN` değerini Colab Secrets veya hücre içi `os.environ[...]` ile verin; token’ı notebook’a yazmayın.
