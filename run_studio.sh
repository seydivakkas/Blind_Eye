#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Blind Eye Studio Launcher (Bash - Linux/macOS)
#  TÜBİTAK 2209-A | Türkçe Dudak Okuma Prototipi
# ═══════════════════════════════════════════════════════════════

set -e

# ANSI Renk Kodları
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Renksiz

echo -e "${CYAN}=======================================================${NC}"
echo -e "${CYAN}  ◉ BLIND EYE STUDIO — BAŞLATICI v1.0                  ${NC}"
echo -e "${CYAN}  Masaüstü Veri Toplama ve Model Eğitim Konsolu        ${NC}"
echo -e "${CYAN}=======================================================${NC}"
echo

# 1. Sanal Ortam Kontrolü
VENV_PATH=".venv"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    VENV_PATH="venv"
fi

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${YELLOW}[BILGI]${NC} Sanal ortam bulunamadı. Yeni bir virtualenv oluşturuluyor (.venv)..."
    python3 -m venv .venv
    VENV_PATH=".venv"
    echo -e "${GREEN}[OK]${NC} Sanal ortam başarıyla oluşturuldu."
fi

# 2. Sanal Ortamı Aktif Et
echo -e "${YELLOW}[BILGI]${NC} Sanal ortam aktif ediliyor ($VENV_PATH)..."
source "$VENV_PATH/bin/activate"

# 3. Bağımlılık Kontrolü ve Yükleme
echo -e "${YELLOW}[BILGI]${NC} Kurulum durumu kontrol ediliyor..."
if ! python3 setup.py --check > /dev/null 2>&1; then
    echo -e "${YELLOW}[UYARI]${NC} Bazı bağımlılıklar eksik görünüyor. Kurulum başlatılıyor..."
    python3 setup.py
else
    echo -e "${GREEN}[OK]${NC} Tüm bağımlılıklar yüklü ve güncel."
fi

# 4. Uygulamayı Başlat
echo
echo -e "${CYAN}=======================================================${NC}"
echo -e "${GREEN}   🚀 Blind Eye Studio başlatılıyor...                 ${NC}"
echo -e "${YELLOW}   (Bu pencereyi kapatmak uygulamayı sonlandırır)      ${NC}"
echo -e "${CYAN}=======================================================${NC}"
echo

python3 studio.py

deactivate
echo -e "${YELLOW}[BILGI]${NC} Kapatıldı."
