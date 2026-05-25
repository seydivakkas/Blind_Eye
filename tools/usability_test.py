"""
tools/usability_test.py
───────────────────────
System Usability Scale (SUS) anketi ve kullanıcı testi framework'ü.

SUS Anketi:
    10 maddelik standart Likert ölçeği (Brooke, 1996)
    Puan aralığı: 0-100 (68+ kabul edilebilir, 85+ mükemmel)

Kullanıcı Senaryoları:
    1. Tek kelime tanıma (doğruluk + süre)
    2. 5 kelimelik dizi testi (WER ölçümü)
    3. 10 dakika kesintisiz kullanım (yorgunluk tespiti)

Kullanım:
    python tools/usability_test.py
    python tools/usability_test.py --participant P01
    python tools/usability_test.py --scenario 1
"""

import os
import sys
import json
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  SUS Anketi (System Usability Scale, Brooke 1996)
# ═══════════════════════════════════════════════════════════════

SUS_QUESTIONS = [
    "Bu sistemi sık sık kullanmak isterim.",
    "Sistemi gereksiz yere karmaşık buldum.",
    "Sistemin kullanımının kolay olduğunu düşünüyorum.",
    "Bu sistemi kullanmak için teknik bir kişinin desteğine ihtiyaç duyarım.",
    "Sistemdeki çeşitli fonksiyonların iyi entegre edildiğini düşünüyorum.",
    "Sistemde çok fazla tutarsızlık olduğunu düşünüyorum.",
    "Çoğu insanın bu sistemi hızlıca öğreneceğini düşünüyorum.",
    "Sistemi kullanırken çok hantal/kullanışsız buldum.",
    "Sistemi kullanırken kendime güveniyordum.",
    "Bu sistemi kullanmadan önce çok şey öğrenmem gerekiyordu."
]

# Tek sorular (1,3,5,7,9) pozitif, çift sorular (2,4,6,8,10) negatif
SUS_POSITIVE_ITEMS = {0, 2, 4, 6, 8}  # 0-indexed


def calculate_sus_score(responses):
    """SUS puanını hesapla.

    Args:
        responses: 10 elemanlı liste, her biri 1-5 arası

    Returns:
        SUS puanı (0-100)
    """
    if len(responses) != 10:
        raise ValueError("SUS anketi 10 yanıt gerektirir!")

    total = 0
    for i, score in enumerate(responses):
        if i in SUS_POSITIVE_ITEMS:
            total += (score - 1)  # Pozitif: puan - 1
        else:
            total += (5 - score)  # Negatif: 5 - puan

    return total * 2.5  # 0-100 ölçeğine normalize et


def interpret_sus_score(score):
    """SUS puanını yorumla."""
    if score >= 85:
        return "Mükemmel (A+)", "🟢"
    elif score >= 73:
        return "İyi (B)", "🟢"
    elif score >= 68:
        return "Kabul Edilebilir (C)", "🟡"
    elif score >= 51:
        return "Marjinal (D)", "🟠"
    else:
        return "Kabul Edilemez (F)", "🔴"


def run_sus_survey(participant_id):
    """Komut satırından SUS anketi uygula."""
    print("\n" + "=" * 60)
    print("  SUS ANKETİ — Blind Eye Dudak Okuma Sistemi")
    print(f"  Katılımcı: {participant_id}")
    print("=" * 60)
    print("\nHer soru için 1-5 arası puan verin:")
    print("  1 = Kesinlikle Katılmıyorum")
    print("  2 = Katılmıyorum")
    print("  3 = Kararsızım")
    print("  4 = Katılıyorum")
    print("  5 = Kesinlikle Katılıyorum")
    print()

    responses = []
    for i, q in enumerate(SUS_QUESTIONS, 1):
        while True:
            try:
                score = int(input(f"  S{i}. {q}\n     Puanınız (1-5): "))
                if 1 <= score <= 5:
                    responses.append(score)
                    break
                else:
                    print("     ⚠️ Lütfen 1-5 arası bir değer girin!")
            except ValueError:
                print("     ⚠️ Lütfen geçerli bir sayı girin!")
            except (EOFError, KeyboardInterrupt):
                print("\n\nAnket iptal edildi.")
                return None

    sus_score = calculate_sus_score(responses)
    label, emoji = interpret_sus_score(sus_score)

    print(f"\n{'='*60}")
    print(f"  {emoji} SUS Puanı: {sus_score:.1f}/100 — {label}")
    print(f"{'='*60}")

    return {
        "participant": participant_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "responses": responses,
        "sus_score": round(sus_score, 1),
        "interpretation": label
    }


# ═══════════════════════════════════════════════════════════════
#  Kullanıcı Test Senaryoları
# ═══════════════════════════════════════════════════════════════

TEST_WORDS = [
    "merhaba", "selam", "gunaydin", "tesekkurederim",
    "hosgeldiniz", "gorusmekuzere", "afiyetolsun",
    "basla", "bitir", "ozurdilerim"
]


def scenario_single_word(participant_id):
    """Senaryo 1: Tek kelime tanıma testi."""
    import random
    target = random.choice(TEST_WORDS)

    print(f"\n{'='*60}")
    print(f"  SENARYO 1: Tek Kelime Tanıma")
    print(f"  Katılımcı: {participant_id}")
    print(f"{'='*60}")
    print(f"\n  Hedef kelime: '{target}'")
    print(f"  Sisteme bu kelimeyi söyleyin ve tahmin süresini ölçeceğiz.")
    print(f"  Hazır olduğunuzda ENTER'a basın...")

    input()
    t_start = time.time()

    prediction = input(f"  Sistemin tahminini girin: ").strip()
    t_end = time.time()

    elapsed = t_end - t_start
    correct = prediction.lower() == target.lower()

    result = {
        "scenario": "single_word",
        "participant": participant_id,
        "target": target,
        "prediction": prediction,
        "correct": correct,
        "response_time_s": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    emoji = "✅" if correct else "❌"
    print(f"\n  {emoji} Sonuç: {'Doğru' if correct else 'Yanlış'} | Süre: {elapsed:.1f}s")

    return result


def scenario_word_sequence(participant_id, num_words=5):
    """Senaryo 2: Kelime dizisi testi."""
    import random
    targets = random.sample(TEST_WORDS, min(num_words, len(TEST_WORDS)))

    print(f"\n{'='*60}")
    print(f"  SENARYO 2: {num_words} Kelimelik Dizi Testi")
    print(f"  Katılımcı: {participant_id}")
    print(f"{'='*60}")

    predictions = []
    correct_count = 0

    for i, target in enumerate(targets, 1):
        print(f"\n  [{i}/{num_words}] Kelime: '{target}'")
        pred = input(f"  Sistemin tahminini girin: ").strip()
        predictions.append(pred)
        if pred.lower() == target.lower():
            correct_count += 1

    accuracy = correct_count / num_words * 100

    result = {
        "scenario": "word_sequence",
        "participant": participant_id,
        "targets": targets,
        "predictions": predictions,
        "accuracy_pct": round(accuracy, 1),
        "correct_count": correct_count,
        "total": num_words,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"\n  Doğruluk: {correct_count}/{num_words} ({accuracy:.0f}%)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Blind Eye Usability Test")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None,
                        help="1: Tek kelime, 2: Kelime dizisi")
    parser.add_argument("--sus-only", action="store_true", help="Sadece SUS anketi")
    parser.add_argument("--output-dir", default="results/usability")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {"participant": args.participant, "tests": []}

    if args.sus_only:
        sus_result = run_sus_survey(args.participant)
        if sus_result:
            results["sus"] = sus_result
    else:
        # SUS anketi
        sus_result = run_sus_survey(args.participant)
        if sus_result:
            results["sus"] = sus_result

        # Senaryolar
        if args.scenario is None or args.scenario == 1:
            s1 = scenario_single_word(args.participant)
            results["tests"].append(s1)

        if args.scenario is None or args.scenario == 2:
            s2 = scenario_word_sequence(args.participant)
            results["tests"].append(s2)

    # Kaydet
    output_path = os.path.join(
        args.output_dir,
        f"usability_{args.participant}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
