# 🐙 Blind Eye — GitHub Dağıtım ve Yayınlama Rehberi

Bu kılavuz, **Blind Eye** projesinin etkileşimli kontrol paneli web sitesini **GitHub Pages** üzerinde yayınlama ve **PyQt6 Studio** arayüzünü GitHub üzerinden temiz bir şekilde dağıtma adımlarını içerir.

---

## 🌐 1. Web Sitesini GitHub Pages ile Yayınlama

Projede yer alan `dashboard/` klasörü, tüm akademik analizleri, ablasyon çalışmalarını, interaktif grafik ve canlandırmaları barındıran **premium** bir statik web sitesidir. 

Bu siteyi yayına almak için hazır bir GitHub Actions iş akışı (CI/CD) entegre edilmiştir.

### 🚀 Otomatik Dağıtım Adımları (GitHub Actions):
1. Projeyi kendi GitHub hesabınızda bir repoya yükleyin (push edin).
2. GitHub üzerinde reponuzun sayfasına gidin: `Settings -> Pages`.
3. **Build and deployment** başlığı altında:
   * **Source** seçeneğini `GitHub Actions` olarak ayarlayın.
4. Repo kök dizininde yer alan `.github/workflows/deploy-pages.yml` dosyası sayesinde, repoya her kod gönderdiğinizde (push) web siteniz otomatik olarak derlenecek ve yayına alınacaktır.
5. Yayındaki web adresiniz şu formatta olacaktır: 
   `https://[username].github.io/Blind_Eye/`

---

## 🖥️ 2. Blind Eye Studio Masaüstü Uygulamasını Dağıtma

**Blind Eye Studio** (`studio.py`), akademik veri toplama ve yapay zeka model eğitim süreçlerini yöneten gelişmiş bir masaüstü kontrol arayüzüdür (PyQt6). 

Geliştiricilerin ve hocalarınızın bu arayüzü tek bir tıklamayla kurup çalıştırabilmesi için platforma özel **başlatıcı (launcher)** script'leri hazırlanmıştır:

### 🪟 Windows Kullanıcıları İçin:
* Proje kök dizininde yer alan **`run_studio.bat`** dosyasına çift tıklayın.
* Bu script otomatik olarak:
  1. `.venv` veya `venv` sanal ortamının varlığını sorgular (yoksa otomatik oluşturur).
  2. Sanal ortamı aktif eder.
  3. Eksik bağımlılıkları `setup.py` aracılığıyla arka planda kurar.
  4. Blind Eye Studio arayüzünü güvenle başlatır.

### 🐧 Linux & macOS Kullanıcıları İçin:
* Terminali açıp proje kök dizininde şu komutları koşturun:
  ```bash
  chmod +x run_studio.sh
  ./run_studio.sh
  ```
* Bu script otomatik olarak sanal ortam kurulumunu tamamlayacak, Python paketlerini yükleyecek ve stüdyoyu çalıştıracaktır.

---

## 📁 3. GitHub İçin Dizin ve Git Süzgeci (.gitignore)

Büyük veri setleri ve ağır PyTorch checkpoint modellerinin repoyu şişirmemesi için `.gitignore` dosyası profesyonelce yapılandırılmıştır:
* **Takip Edilmeyenler (Ignored):** Büyük model ağırlıkları (`*.pth`), raw veri setleri (`data/raw/` altındaki devasa .zip dosyaları) ve geçici cache dizinleri (`__pycache__`, `.pytest_cache`).
* **Takip Edilenler (Tracked):** Tüm kaynak kodlar (`backend`, `frontend`, `pc`, `pi`, `ui`), test senaryoları, konfigürasyon dosyaları, örnek metrik logları ve sıkıştırılmış hafif ONNX modelleri (`student_int8.onnx` ~800KB).

---

## 📊 4. Web Kontrol Paneli Sayfaları (Dashboard)

Kullanıcılar veya jüriler tarayıcı üzerinden yayındaki sitenize bağlandığında şu etkileşimli sekmelere erişebilirler:
1. **Genel Bakış (Overview):** 29 backend modülünün haritası, performans trendleri ve WER / CER genel KPI kartları.
2. **Pipeline Durumu:** Kameradan ROI çıkarmaya ve CTC decoder'a kadar çalışan uçtan uca animasyonlu yapay zeka pipeline takibi.
3. **Eğitim & Model:** Ablasyon deneyi tabloları, hiperparametre ayarlama kaydırıcıları ve eğitim kayıpları grafiği.
4. **Analiz Bölümü:** Türkçe fonem karışıklık matrisi (Confusion Matrix) ve LOSO cross-validation sonuçları.
5. **Akademik:** Türkçe viseme grupları, fonotaktik geçiş matrisleri ve model belirsizlik (Uncertainty) metrikleri.
6. **Klinik & XAI:** Açıklanabilir yapay zeka (XAI) attention haritası, NASA-TLX bilişsel yük anket sonuçları ve kullanılabilirlik skorları.
7. **Dağıtım (Deploy):** Sıkıştırılmış hafif ONNX modellerinin export butonları ve Pi 3 B+ donanım telemetri paneli.
