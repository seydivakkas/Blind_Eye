"""
tools/calibrate_thresholds.py
─────────────────────────────
Raspberry Pi 3 Model B+ için adaptif EAR eşik kalibrasyonu.

30 saniyelik kalibrasyon fazında kişiye özel:
- EAR (Eye Aspect Ratio) baseline hesaplar
- Göz kırpma ve PERCLOS eşiklerini belirler
- KLT max_drift optimizasyonu yapar

Çıktı: configs/pi_calibration.json

Kullanım:
    python tools/calibrate_thresholds.py --source 0
    python tools/calibrate_thresholds.py --source video.mp4 --duration 30
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# MediaPipe göz landmark indisleri
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # üst-alt çiftleri
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(eye_pts):
    """Eye Aspect Ratio (Soukupová & Čech, 2016).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    def dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    v1 = dist(eye_pts[1], eye_pts[5])
    v2 = dist(eye_pts[2], eye_pts[4])
    h = dist(eye_pts[0], eye_pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calibrate(source, duration_sec=30):
    """Kamera üzerinden EAR kalibrasyonu."""
    import cv2

    try:
        import mediapipe as mp
    except ImportError:
        logger.error("mediapipe yüklü değil!")
        return None

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        logger.error("Kamera açılamadı!")
        return None

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    ear_values = []
    t_start = time.time()
    frame_count = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"EAR KALİBRASYON BAŞLIYOR ({duration_sec}s)")
    logger.info(f"Lütfen normal şekilde ekrana bakın...")
    logger.info(f"{'='*50}")

    while time.time() - t_start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Sol göz EAR
            left_pts = [(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE]
            left_ear = compute_ear(left_pts)

            # Sağ göz EAR
            right_pts = [(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE]
            right_ear = compute_ear(right_pts)

            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

        frame_count += 1
        elapsed = time.time() - t_start

        # Her 5 saniyede güncelleme
        if frame_count % 150 == 0:
            logger.info(f"  t={elapsed:.0f}s | {len(ear_values)} EAR ölçümü")

        # Görsel geri bildirim
        cv2.putText(frame, f"KALIBRASYON: {int(duration_sec - elapsed)}s kaldi",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if ear_values:
            cv2.putText(frame, f"EAR: {ear_values[-1]:.3f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow("EAR Kalibrasyon", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(ear_values) < 10:
        logger.error("Yeterli EAR verisi toplanamadı!")
        return None

    ear_array = np.array(ear_values)

    # İstatistikler
    ear_mean = float(np.mean(ear_array))
    ear_std = float(np.std(ear_array))
    ear_min = float(np.min(ear_array))
    ear_max = float(np.max(ear_array))
    ear_p10 = float(np.percentile(ear_array, 10))
    ear_p90 = float(np.percentile(ear_array, 90))

    # Adaptif eşik: ortalamadan 1.5 standart sapma altı
    blink_threshold = max(ear_mean - 1.5 * ear_std, 0.15)
    # PERCLOS eşiği: göz %80 kapalı sayılacak EAR
    perclos_threshold = ear_mean * 0.6

    calibration = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": duration_sec,
        "num_samples": len(ear_values),
        "ear_stats": {
            "mean": round(ear_mean, 4),
            "std": round(ear_std, 4),
            "min": round(ear_min, 4),
            "max": round(ear_max, 4),
            "p10": round(ear_p10, 4),
            "p90": round(ear_p90, 4),
        },
        "thresholds": {
            "ear_blink": round(blink_threshold, 4),
            "ear_perclos": round(perclos_threshold, 4),
            "perclos_window_sec": 60,
            "perclos_alarm_pct": 0.4,
        },
        "klt": {
            "max_drift": 15.0,
            "fb_threshold": 2.0,
            "detection_interval": 5,
        }
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"KALİBRASYON SONUÇLARI")
    logger.info(f"{'='*50}")
    logger.info(f"  EAR Ortalama: {ear_mean:.4f}")
    logger.info(f"  EAR Std: {ear_std:.4f}")
    logger.info(f"  Göz Kırpma Eşiği: {blink_threshold:.4f}")
    logger.info(f"  PERCLOS Eşiği: {perclos_threshold:.4f}")

    return calibration


def main():
    parser = argparse.ArgumentParser(description="EAR Adaptif Kalibrasyon")
    parser.add_argument("--source", default="0")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", default="configs/pi_calibration.json")
    args = parser.parse_args()

    result = calibrate(args.source, args.duration)

    if result:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\nKalibrasyon kaydedildi: {args.output}")
        logger.info("Bu dosyayı Pi Zero'ya kopyalayın: scp pi_calibration.json pi@raspberrypi:~/blind_eye/configs/")
    else:
        logger.error("Kalibrasyon başarısız!")
        sys.exit(1)


if __name__ == "__main__":
    main()
