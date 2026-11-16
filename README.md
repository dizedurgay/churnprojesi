# Telekom Müşteri Kaybı (Churn) Tahmini

Bu proje, IBM Telco Customer Churn veri seti üzerinde ikili sınıflandırma modeli kurar. Amaç, müşterinin hizmetten ayrılıp ayrılmayacağını (`Churn`) tahmin etmektir.

## İçerik
- `src/train.py`: veri yükleme, EDA, ön işleme, eğitim, değerlendirme ve çıktı üretimi
- `src/make_report.py`: `outputs/metrics.json` içeriğini kullanarak Word raporu üretir
- `outputs/figures/`: görseller
- `outputs/metrics.json`: metrikler ve gözlemler

## Çalıştırma
```bash
python src/train.py
python src/make_report.py
```

## Notlar
- Veri kaynağı: IBM Telco Customer Churn
- `TotalCharges` alanı metin formatındadır; sayısal formata dönüştürülür ve boş değerler doldurulur.
- `customerID` sütunu modele alınmaz.
- Test bölünmesi %80 train / %20 test olup `stratify=y` ile yapılır.

## Sonuç Özeti
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8416 |
| Random Forest | 0.7587 | 0.5309 | 0.7807 | 0.6320 | 0.8428 |

İlk çalıştırma sonrası `outputs/metrics.json` dosyasından güncel değerler alınır. Random Forest biraz daha yüksek accuracy ve F1 skoruna sahiptir; Logistic Regression ise daha yüksek recall sunar.
