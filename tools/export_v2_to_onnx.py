"""
tools/export_v2_to_onnx.py
──────────────────────────
TÜBİTAK 2209-A — LipReadModelV2 (ResNet-18 + CBAM + Conformer) PyTorch → ONNX

Kullanım:
    python tools/export_v2_to_onnx.py --checkpoint models/checkpoints/v2_best.pth --output models/pi_model_float32.onnx
"""

import os
import sys
import json
import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_v2_onnx")

def main():
    parser = argparse.ArgumentParser(description="LipReadModelV2 ONNX Exporter")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/v2_best.pth",
                        help="Giriş PyTorch checkpoint (.pth) dosyası")
    parser.add_argument("--output", type=str, default="models/pi_model_float32.onnx",
                        help="Çıktı ONNX dosya yolu")
    parser.add_argument("--d-model", type=int, default=256, help="Conformer d_model boyutu")
    parser.add_argument("--n-layers", type=int, default=4, help="Conformer katman sayısı")
    args = parser.parse_args()

    # ── 1. Vocab Yükle ──
    vocab_path = "configs/vocab.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            num_classes = json.load(f).get("num_classes", 31)
    else:
        num_classes = 31
        logger.warning(f"vocab.json bulunamadı! Varsayılan num_classes={num_classes} kullanılacak.")

    # ── 2. Model Kurulumu ──
    from backend.cbam import LipReadModelV2
    logger.info(f"Model V2 kuruluyor (num_classes={num_classes}, d_model={args.d_model})...")
    model = LipReadModelV2(
        num_classes=num_classes,
        d_model=args.d_model,
        n_conf_layers=args.n_layers
    )

    # ── 3. Checkpoint Ağırlıklarını Yükle ──
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint bulunamadı: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Checkpoint yükleniyor: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # "model_state" veya direkt state_dict çözümü
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # ── 4. Sahte Girdi Hazırla ──
    # [Batch=1, Time=30, Height=96, Width=96, Channels=1]
    dummy_input = torch.randn(1, 30, 96, 96, 1)

    # ── 5. ONNX Aktarımı (TorchScript Tracing ile Bypass) ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logger.info(f"ONNX export başlıyor → {args.output}")

    try:
        # PyTorch 2.X Dynamo exporter hatalarını aşmak için JIT trace uyguluyoruz
        logger.info("Model JIT tracing uygulanıyor (bu sayede conformer temporal boyutları dondurulur)...")
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        
        torch.onnx.export(
            traced_model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch", 1: "time"},
                "output": {0: "batch", 1: "time"},
            }
        )
        logger.info(f"✅ ONNX modeli başarıyla oluşturuldu!")
    except Exception as e:
        logger.error(f"JIT Traced ONNX export başarısız, standart legacy export deneniyor: {e}")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                args.output,
                export_params=True,
                opset_version=15,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input":  {0: "batch", 1: "time"},
                    "output": {0: "batch", 1: "time"},
                }
            )
            logger.info(f"✅ Standart ONNX modeli başarıyla oluşturuldu!")
        except Exception as e2:
            logger.error(f"ONNX Export tamamen başarısız: {e2}")
            sys.exit(1)

    # ── 6. Doğrulama (ONNX Runtime) ──
    if os.path.exists(args.output):
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            # Boyut kontrolü
            size_mb = os.path.getsize(args.output) / (1024 ** 2)
            logger.info(f"ONNX Model Boyutu: {size_mb:.2f} MB")

            # Check model graph integrity
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX grafiği doğrulandı!")

            # ONNX Runtime ile test çıkarımı (seq_len = 6, gözlük simülasyonu)
            sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
            test_inp = np.random.randn(1, 6, 96, 96, 1).astype(np.float32)
            out = sess.run(None, {"input": test_inp})[0]
            logger.info(f"✅ ONNX Doğrulama Başarılı! Çıktı Boyutu: {out.shape} (beklenen: [1, 6, {num_classes}])")
        except Exception as e_verify:
            logger.warning(f"ONNX doğrulama sırasında uyarı alındı: {e_verify}")

if __name__ == "__main__":
    main()
