"""Mock pipeline end-to-end testi."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc.preprocess import Preprocessor
from pc.vsr_engine import VSREngine
from pc.face_cues import FaceCueAnalyzer
from pc.fusion_decoder import FusionDecoder

print("=== Blind Eye Mock Pipeline Testi ===")

# 1. VSR Engine
engine = VSREngine(model_path="models/student_fp32.onnx")
print(f"[1] VSR Engine: provider={engine.active_provider}, T={engine.chunk_size}, V={engine.num_classes}")

# 2. Face Cue Analyzer
cues = FaceCueAnalyzer(fps=15.0)
print("[2] FaceCueAnalyzer olusturuldu")

# 3. Fusion Decoder
decoder = FusionDecoder(lm_path=None, beam_width=50)
print("[3] FusionDecoder olusturuldu")

# 4. Mock inference
dummy_roi = np.random.rand(engine.chunk_size, 96, 96, 1).astype(np.float32)
logits = engine.infer(dummy_roi)
print(f"[4] Logits shape: {logits.shape}, latency: {engine.last_latency_ms:.1f}ms")

# 5. Mock decode
result = decoder.decode(logits, None)
print(f"[5] Decoded: \"{result.text}\" (conf={result.confidence:.2f}, lat={result.latency_ms:.1f}ms)")

# 6. Face cues (sintetik landmark)
fake_landmarks = [(100 + i, 200 + i) for i in range(478)]
cue = cues.analyze(fake_landmarks)
print(f"[6] FaceCue: blink={cue.is_blinking}, brow={cue.eyebrow_raised}, nod={cue.head_nod_detected}")

# 7. Decode with face cue
result2 = decoder.decode(logits, cue)
print(f"[7] Fused: \"{result2.text}\" (cue_applied={result2.face_cue_applied})")

# 8. Subtitle lines
l1, l2 = decoder.get_subtitle_lines()
print(f"[8] Subtitle: \"{l1}\" | \"{l2}\"")

print("\n=== TAMAMLANDI: Tum moduller basariyla calisiyor ===")
