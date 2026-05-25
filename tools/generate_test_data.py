"""
tools/generate_test_data.py
Pipeline doğrulama için sentetik test videoları üretir.

Mendeley dataset indirilene kadar tüm araçları test etmek için kullanılır.
Gerçek dudak hareketlerini simüle eden basit OpenCV videolar oluşturur.
"""

import os
import sys
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 10 kelimelik test seti (Mendeley veri setinden esinlenilmiş)
TEST_WORDS = [
    "merhaba",
    "evet",
    "hayir",
    "basla",
    "durdur",
    "tesekkurler",
    "gunaydin",
    "hosgeldiniz",
    "tamam",
    "lutfen",
]


def generate_lip_video(
    output_path: str,
    word: str,
    num_frames: int = 30,
    size: tuple = (320, 240),
    fps: int = 25,
):
    """Sentetik dudak hareketi videosu oluşturur.

    Basit elips animasyonu ile dudak açılma/kapanma simülasyonu.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        # Yüz simülasyonu (basit oval)
        cx, cy = size[0] // 2, size[1] // 2
        cv2.ellipse(frame, (cx, cy - 10), (80, 100), 0, 0, 360, (180, 140, 100), -1)

        # Gözler
        cv2.circle(frame, (cx - 30, cy - 40), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 30, cy - 40), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 30, cy - 40), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 30, cy - 40), 4, (50, 50, 50), -1)

        # Dudak animasyonu — sinüsoidal açılma/kapanma
        progress = i / max(num_frames - 1, 1)
        mouth_open = abs(np.sin(progress * np.pi * 2 + hash(word) % 10)) * 15 + 3

        # Üst dudak
        cv2.ellipse(frame, (cx, cy + 30), (25, int(mouth_open)),
                     0, 180, 360, (150, 80, 80), -1)
        # Alt dudak
        cv2.ellipse(frame, (cx, cy + 30), (25, int(mouth_open)),
                     0, 0, 180, (140, 70, 70), -1)

        # Dudak konturu
        cv2.ellipse(frame, (cx, cy + 30), (25, int(mouth_open)),
                     0, 0, 360, (120, 50, 50), 2)

        # Rastgele gürültü (gerçekçilik)
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        writer.write(frame)

    writer.release()


def generate_dataset(
    output_dir: str = "data/raw",
    words: list = None,
    videos_per_word: int = 5,
    frames_per_video: int = 30,
):
    """Tam test veri seti oluşturur."""
    words = words or TEST_WORDS

    logger.info(f"🎬 Sentetik veri seti oluşturuluyor...")
    logger.info(f"   Kelimeler: {len(words)}")
    logger.info(f"   Video/kelime: {videos_per_word}")
    logger.info(f"   Frame/video: {frames_per_video}")

    total = 0
    for word in words:
        word_dir = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)

        for j in range(videos_per_word):
            filename = f"{word}_{j+1:02d}.mp4"
            filepath = os.path.join(word_dir, filename)

            # Her video biraz farklı olsun
            n_frames = frames_per_video + np.random.randint(-5, 6)
            generate_lip_video(filepath, word, num_frames=max(15, n_frames))
            total += 1

    logger.info(f"\n✅ {total} video oluşturuldu → {output_dir}")
    logger.info(f"   Toplam: {len(words)} kelime × {videos_per_word} video")

    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentetik Test Veri Seti Üretici")
    parser.add_argument("--output", default="data/raw",
                        help="Çıktı klasörü")
    parser.add_argument("--words", type=int, default=10,
                        help="Kelime sayısı (max 10)")
    parser.add_argument("--videos", type=int, default=5,
                        help="Kelime başına video sayısı")
    parser.add_argument("--frames", type=int, default=30,
                        help="Video başına frame sayısı")

    args = parser.parse_args()

    words = TEST_WORDS[:args.words]
    generate_dataset(
        output_dir=args.output,
        words=words,
        videos_per_word=args.videos,
        frames_per_video=args.frames,
    )
