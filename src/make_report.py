import json
import os
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"


def load_metrics():
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_h2(doc, title):
    p = doc.add_paragraph()
    p.add_run(title).bold = True


def add_section_paragraph(doc, text):
    doc.add_paragraph(text)


def add_image(doc, image_path: Path):
    if image_path.exists():
        doc.add_picture(str(image_path), width=Inches(5.5))


def main():
    metrics = load_metrics()
    repo_url = os.environ.get("REPO_URL", "GITHUB_LINK_BURAYA")
    document = Document()
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(11)

    title = "Telekom Müşteri Kaybı (Churn) Tahmini: Logistic Regression ve Random Forest Karşılaştırması"
    h = document.add_heading(title, level=0)
    h.alignment = 1

    add_h2(document, "1. Problem Tanımı")
    document.add_paragraph(
        "Bu çalışma, telekom müşterilerinin hizmetten ayrılıp ayrılmayacağını tahmin etmeyi amaçlar. "
        "Hedef değişken Churn olup, müşteri kaybı riskini öngörmek için operasyonel ve müşteri profiline ilişkin özellikler kullanılır. "
        "Bu model, müşteri ilişkileri yöneticilerine erken müdahale edebilmek ve kayıp oranını azaltmak için kullanılabilir."
    )

    add_h2(document, "2. Veri Seti")
    document.add_paragraph(
        f"Veri kaynağı: {metrics['source']}. Toplam örnek sayısı {metrics['dataset']['rows']} ve hedef değişken Churn'dir. "
        f"Veri setinde {metrics['dataset']['columns']} sütun bulunmaktadır. Sınıf dağılımı; No: {metrics['dataset']['class_balance']['No']}, Yes: {metrics['dataset']['class_balance']['Yes']} şeklindedir. "
        "Sınıf dengesizliği nedeniyle test bölünmesinde stratify parametresi kullanılmıştır."
    )

    add_h2(document, "3. Veri Analizi ve Ön İşleme")
    for obs in metrics["observations"]:
        document.add_paragraph(f"{obs['title']}: {obs['text']}")

    document.add_paragraph(
        "Ön işleme aşamasında, TotalCharges alanı numeric formata çevrilmiş, hatalı değerler 0 ile doldurulmuştur. "
        "Müşteri kimlik sütunu modele dahil edilmemiştir. Kategorik sütunlar One-Hot kodlama ile dönüştürülürken, sayısal sütunlar StandardScaler ile ölçeklendirilmiştir. "
        "Bütün bu işlemler ColumnTransformer ve Pipeline yapısı içinde uygulanmıştır."
    )

    add_image(document, OUTPUTS_DIR / "figures" / "01_churn_distribution.png")
    add_image(document, OUTPUTS_DIR / "figures" / "02_contract_churn_rate.png")
    add_image(document, OUTPUTS_DIR / "figures" / "03_tenure_monthlycharges.png")

    add_h2(document, "4. Model / Yöntem")
    document.add_paragraph(
        "Logistic Regression ve Random Forest modelleri aynı test setinde karşılaştırılmıştır. Logistic Regression, doğrusal ayrımın güçlü olduğu durumlarda yorumlanabilir bir temel model olarak seçilmiştir. "
        "Random Forest ise non-linear ilişkileri yakalama potansiyeli nedeniyle tercih edilmiştir. LR için max_iter=1000 ve class_weight='balanced' kullanılırken, RF için 5 katlı GridSearchCV ile n_estimators, max_depth ve min_samples_leaf parametreleri optimize edilmiştir."
    )

    add_h2(document, "5. Eğitim / Uygulama Süreci")
    lr_cv = metrics["cross_validation"]["logistic_regression_f1_mean"]
    rf_cv = metrics["cross_validation"]["random_forest_best_cv_f1"]
    document.add_paragraph(
        f"Veri %80 eğitim / %20 test şeklinde ayrılmıştır. Logistic Regression için 5-katlı çapraz doğrulama F1 skoru {lr_cv:.4f}, Random Forest için en iyi çapraz doğrulama F1 skoru {rf_cv:.4f} olarak elde edilmiştir. "
        "Bu süreçte random_state=42 sabit tutulmuştur."
    )

    add_h2(document, "6. Değerlendirme Sonuçları")
    lr = metrics["models"]["logistic_regression"]["test_metrics"]
    rf = metrics["models"]["random_forest"]["test_metrics"]
    comparison = metrics["comparison_table"]
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1"]
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
    for row_data in comparison:
        cells = table.add_row().cells
        cells[0].text = row_data["model"]
        cells[1].text = f"{row_data['accuracy']:.4f}"
        cells[2].text = f"{row_data['precision']:.4f}"
        cells[3].text = f"{row_data['recall']:.4f}"
        cells[4].text = f"{row_data['f1']:.4f}"
        cells[5].text = f"{row_data['roc_auc']:.4f}"
        cells[6].text = f"{row_data.get('cv_f1_mean', row_data.get('best_cv_f1')):.4f}"

    document.add_paragraph(
        f"Logistic Regression için accuracy {lr['accuracy']:.4f}, precision {lr['precision']:.4f}, recall {lr['recall']:.4f}, F1 {lr['f1']:.4f}, ROC-AUC {lr['roc_auc']:.4f}. "
        f"Random Forest için accuracy {rf['accuracy']:.4f}, precision {rf['precision']:.4f}, recall {rf['recall']:.4f}, F1 {rf['f1']:.4f}, ROC-AUC {rf['roc_auc']:.4f}."
    )
    add_image(document, OUTPUTS_DIR / "figures" / "06_lr_confusion_matrix.png")
    add_image(document, OUTPUTS_DIR / "figures" / "07_rf_confusion_matrix.png")
    add_image(document, OUTPUTS_DIR / "figures" / "08_roc_curve.png")
    add_image(document, OUTPUTS_DIR / "figures" / "05_feature_importance_rf.png")

    add_h2(document, "7. Hata Analizi")
    if lr["recall"] >= rf["recall"]:
        best_model = "Logistic Regression"
    else:
        best_model = "Random Forest"
    document.add_paragraph(
        f"Churn sınıfı için recall değeri kritik bir metriktir çünkü kaçan müşterileri kaçırmamak gerekir. Bu bağlamda {best_model} modeli, daha yüksek recall sağlama potansiyeli nedeniyle tercih edilebilir. "
        "Ancak sınıf dengesizliği, düşük churn oranı ve eşik değeri 0.5 nedeniyle yanlış negatifler oluşabilmektedir."
    )
    document.add_paragraph(
        "False Negative örneklerinde genellikle kısa tenure ve aylık sözleşme yapısı öne çıkmaktadır. Ayrıca modelin karar eşiği 0.5 olarak sabit tutulduğundan, daha yüksek recall için eşiğin ayarlanması avantaj sağlayabilir."
    )

    add_h2(document, "8. Sonuç ve Geliştirme Önerileri")
    document.add_paragraph(
        "Sonuç olarak, her iki model de churn tahmini için uygulanabilir olsa da, elde edilen performans ve dikkat edilmesi gereken metrikler doğrultusunda seçilen yaklaşım, gerçek hayatta müşteri kaybını önlemeye odaklanmalıdır. "
        "Geliştirme önerileri arasında eşik ayarı, SMOTE ile dengesizlik giderme, XGBoost/LightGBM kullanımı ve maliyete duyarlı değerlendirme yer almaktadır."
    )

    document.add_paragraph("Kod: " + repo_url)
    document.add_paragraph("Öğrenci adı: Dize ...")
    document.save(OUTPUTS_DIR / "rapor.docx")
    print("Rapor oluşturuldu: outputs/rapor.docx")


if __name__ == "__main__":
    main()
