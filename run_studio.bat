@echo off
:: ═══════════════════════════════════════════════════════════════
#  Blind Eye Studio Launcher (Windows)
#  TÜBİTAK 2209-A | Türkçe Dudak Okuma Prototipi
:: ═══════════════════════════════════════════════════════════════
chcp 65001 > nul
setlocal enabledelayedexpansion

echo =======================================================
echo   ◉ BLIND EYE STUDIO — BAŞLATICI v1.0
echo   Masaüstü Veri Toplama ve Model Eğitim Konsolu
echo =======================================================
echo.

:: 1. Sanal Ortam Kontrolü
set VENV_PATH=.venv
if not exist !VENV_PATH!\Scripts\activate.bat (
    set VENV_PATH=venv
)

if not exist !VENV_PATH!\Scripts\activate.bat (
    echo [BILGI] Sanal ortam bulunamadı. Yeni bir virtualenv oluşturuluyor (.venv)...
    python -m venv .venv
    set VENV_PATH=.venv
    if !errorlevel! neq 0 (
        echo [HATA] Python sanal ortamı oluşturulamadı! Lütfen Python'ın PATH'e ekli olduğunu kontrol edin.
        pause
        exit /b 1
    )
    echo [OK] Sanal ortam başarıyla oluşturuldu.
)

:: 2. Sanal Ortamı Aktif Et
echo [BILGI] Sanal ortam aktif ediliyor (!VENV_PATH!)...
call !VENV_PATH!\Scripts\activate.bat

:: 3. Bağımlılık Kontrolü ve Yükleme
echo [BILGI] Kurulum durumu kontrol ediliyor...
python setup.py --check > nul 2>&1
if !errorlevel! neq 0 (
    echo [UYARI] Bazı bağımlılıklar eksik görünüyor. Kurulum başlatılıyor...
    python setup.py
    if !errorlevel! neq 0 (
        echo [HATA] Bağımlılıkların yüklenmesi başarısız oldu!
        pause
        exit /b 1
    )
) else (
    echo [OK] Tüm bağımlılıklar yüklü ve güncel.
)

:: 4. Uygulamayı Başlat
echo.
echo =======================================================
echo   🚀 Blind Eye Studio başlatılıyor...
echo   (Bu pencereyi kapatmak uygulamayı sonlandırır)
echo =======================================================
echo.

python studio.py

if !errorlevel! neq 0 (
    echo.
    echo [HATA] Blind Eye Studio beklenmedik bir şekilde sonlandı!
    echo Hata Ayrıntıları için yukarıdaki logları inceleyebilirsiniz.
    pause
)

deactivate
echo [BILGI] Kapatıldı.
