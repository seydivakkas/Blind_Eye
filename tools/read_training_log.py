import json, os
log_path = "results/training_log.json"
if not os.path.exists(log_path):
    print("Log henuz yok.")
else:
    with open(log_path) as f:
        log = json.load(f)
    if not log:
        print("Log bos.")
    else:
        last = log[-1]
        best = min(log, key=lambda x: x["val_loss"])
        print(f"Kayitli epoch sayisi : {len(log)} / 50")
        print(f"Son epoch  : E{last['epoch']:02d}  train={last['train_loss']:.4f}  val={last['val_loss']:.4f}  WER={last['wer']*100:.1f}%  CER={last['cer']*100:.1f}%")
        print(f"En iyi     : E{best['epoch']:02d}  val={best['val_loss']:.4f}  WER={best['wer']*100:.1f}%  CER={best['cer']*100:.1f}%")
        print()
        print("Epoch  | val_loss | WER%   | CER%")
        print("-"*40)
        for e in log:
            marker = " <-- best" if e["epoch"] == best["epoch"] else ""
            print(f"  E{e['epoch']:02d}   |  {e['val_loss']:.4f}  | {e['wer']*100:5.1f}% | {e['cer']*100:5.1f}%{marker}")
