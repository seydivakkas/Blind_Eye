"""Export best checkpoint to ONNX (PyTorch 2.11 uyumlu)."""
import sys, os, json, torch
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
from backend.cbam import LipReadModelWithCBAM

vocab = json.load(open("configs/vocab.json"))
num_classes = vocab["num_classes"]

model = LipReadModelWithCBAM(num_classes=num_classes, hidden_dim=128)
ck = torch.load("models/checkpoints/best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state"])
model.eval()

dummy = torch.randn(1, 30, 96, 96, 1)
out_path = "models/student_fp32.onnx"

# PyTorch 2.11 yeni exporter sorun cikariyor, dynamo_export veya eski API dene
try:
    # Eski API (legacy) ile dene
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Yeni dynamo exporter yerine klasik kullan
    )
except TypeError:
    # dynamo parametresi desteklenmiyorsa, dogrudan TorchScript
    print("Fallback: TorchScript ile export...")
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, "models/student_fp32.pt")
    # Sonra ONNX cevir
    torch.onnx.export(
        traced, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

if os.path.exists(out_path):
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    epoch = ck["epoch"]
    val_loss = ck["val_loss"]
    print(f"ONNX kaydedildi: {out_path} ({size_mb:.1f} MB)")
    print(f"Checkpoint epoch: {epoch}, val_loss: {val_loss:.4f}")

    # ONNX dogrulama
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(out_path)
    inp = np.random.randn(1, 30, 96, 96, 1).astype(np.float32)
    out = sess.run(None, {"input": inp})[0]
    print(f"ONNX cikti shape: {out.shape}  (beklenen: [1, 30, {num_classes}])")
    print("ONNX dogrulama basarili!")
else:
    print("ONNX olusturulamadi.")
