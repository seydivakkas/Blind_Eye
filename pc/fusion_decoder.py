"""
pc/fusion_decoder.py
════════════════════
VSR Logits + Face Cues → pyctcdecode + KenLM (TR 3-gram) Fusion Decoder

İki bilgi kaynağını birleştirir:
    1. VSR logits [1, T, 31] — ResNet18 + DC-TCN + CTC model çıktısı
    2. Face cues — kaş/göz/baş → punctuation sinyalleri

Decode pipeline:
    logits → softmax → pyctcdecode beam search (KenLM 3-gram)
    → face cue post-processing (?, ., , ekleme)
    → final text + confidence

Kullanım:
    decoder = FusionDecoder(lm_path="models/tr_3gram.arpa")
    result = decoder.decode(logits, face_cue)
    print(result.text, result.confidence)

Referans:
    - pyctcdecode: CTC beam search with language model
    - KenLM: Kenneth Heafield's language model toolkit
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# pyctcdecode import
try:
    from pyctcdecode import build_ctcdecoder
    _PYCTC_AVAILABLE = True
except ImportError:
    _PYCTC_AVAILABLE = False
    logger.info("pyctcdecode yüklü değil — greedy decoder kullanılacak.")

# Face cues import (opsiyonel — olmadan da çalışır)
try:
    from .face_cues import FaceCueResult
except ImportError:
    FaceCueResult = None

# Vocab dosyası
_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs", "vocab.json"
)


def _load_vocab() -> List[str]:
    """configs/vocab.json'dan karakter listesini yükler."""
    try:
        with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["charset"]
    except Exception:
        # Fallback: Türkçe karakter seti
        return ["<blank>"] + list("abcçdefgğhıijklmnoöprsştuüvyz ")


@dataclass
class DecodeResult:
    """Fusion decoder çıktısı."""
    text: str = ""                    # Decode edilmiş metin
    raw_text: str = ""                # Punctuation öncesi ham metin
    confidence: float = 0.0           # VSR güven skoru (0–1)
    punctuation: str = ""             # Eklenen punctuation karakteri
    face_cue_applied: bool = False    # Face cue uygulandı mı?
    latency_ms: float = 0.0          # Decode süresi


class FusionDecoder:
    """VSR + Face Cues Fusion Decoder.

    Parameters
    ----------
    lm_path : str | None
        KenLM .arpa model yolu. None ise LM'siz beam search.
    alpha : float
        Dil modeli ağırlığı (0.0–2.0).
    beta : float
        Kelime ekleme bonusu.
    beam_width : int
        Beam search genişliği.
    vocab : list[str] | None
        Karakter seti. None ise vocab.json'dan yükler.
    min_confidence_for_cue : float
        Face cue uygulamak için minimum VSR güveni.
    """

    def __init__(
        self,
        lm_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        beam_width: int = 100,
        vocab: Optional[List[str]] = None,
        min_confidence_for_cue: float = 0.3,
    ):
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.vocab = vocab or _load_vocab()
        self.min_confidence_for_cue = min_confidence_for_cue

        self._decoder = None
        self._use_beam = False

        # Regex temizleme (Türkçe)
        self._patterns = [
            (re.compile(r"(.)\1{2,}"), r"\1"),           # 3+ tekrar → 1
            (re.compile(r"[^\w\s.,!?'\-]"), ""),          # Özel karakter temizle
            (re.compile(r"\s{2,}"), " "),                 # Çoklu boşluk → tek
        ]

        # Son decode sonucu (subtitle oluşturma için)
        self._subtitle_buffer: list = []
        self._max_subtitle_lines = 2

        # ── pyctcdecode beam search decoder ──
        if _PYCTC_AVAILABLE:
            try:
                labels = self._prepare_labels()

                if lm_path and os.path.exists(lm_path):
                    self._decoder = build_ctcdecoder(
                        labels=labels,
                        kenlm_model_path=lm_path,
                        alpha=alpha,
                        beta=beta,
                    )
                    self._use_beam = True
                    logger.info(
                        f"FusionDecoder hazır (KenLM): {lm_path} "
                        f"(α={alpha}, β={beta}, beam={beam_width})"
                    )
                else:
                    self._decoder = build_ctcdecoder(labels=labels)
                    self._use_beam = True
                    logger.info("FusionDecoder hazır (Beam Search, LM yok)")

            except Exception as e:
                logger.warning(f"pyctcdecode başlatılamadı: {e}")
                self._use_beam = False

        if not self._use_beam:
            logger.info("FusionDecoder: Greedy CTC Decoder aktif (fallback)")

    def decode(
        self,
        logits: np.ndarray,
        face_cue: Optional[object] = None,
    ) -> DecodeResult:
        """CTC logits + face cues → metin.

        Parameters
        ----------
        logits : np.ndarray
            [1, T, V] veya [T, V] boyutlu CTC logits.
        face_cue : FaceCueResult | None
            Face cue analiz sonucu. None ise sadece VSR decode.

        Returns
        -------
        DecodeResult
            Decode edilmiş metin, güven, punctuation bilgisi.
        """
        t0 = time.perf_counter()
        result = DecodeResult()

        if logits is None:
            return result

        # Squeeze batch dimension
        if logits.ndim == 3:
            logits = logits.squeeze(0)  # [T, V]

        # ── CTC Decode ──
        if self._use_beam:
            raw_text, confidence = self._beam_decode(logits)
        else:
            raw_text, confidence = self._greedy_decode(logits)

        result.raw_text = raw_text
        result.confidence = confidence

        # ── Regex Temizleme ──
        text = raw_text
        for pattern, repl in self._patterns:
            text = pattern.sub(repl, text)
        text = text.strip()

        # ── Face Cue Post-Processing ──
        if (
            face_cue is not None
            and FaceCueResult is not None
            and isinstance(face_cue, FaceCueResult)
            and confidence >= self.min_confidence_for_cue
            and text
        ):
            text, punct = self._apply_face_cues(text, face_cue)
            result.punctuation = punct
            result.face_cue_applied = bool(punct)

        result.text = text

        # ── Subtitle Buffer Güncelle ──
        if text:
            self._update_subtitle_buffer(text)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def get_subtitle_lines(self) -> Tuple[str, str]:
        """OLED için 2 satırlık altyazı döndür.

        Returns
        -------
        (line1, line2) : tuple[str, str]
            line1 = son cümle, line2 = önceki cümle
        """
        if len(self._subtitle_buffer) >= 2:
            return self._subtitle_buffer[-1], self._subtitle_buffer[-2]
        elif len(self._subtitle_buffer) == 1:
            return self._subtitle_buffer[-1], ""
        return "", ""

    # ──────────── DECODE METHODS ────────────

    def _beam_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """pyctcdecode beam search."""
        try:
            text = self._decoder.decode(logits, beam_width=self.beam_width)
            probs = self._softmax(logits)
            conf = float(np.mean(np.max(probs, axis=-1)))
            return text.strip(), max(min(conf, 1.0), 0.0)
        except Exception as e:
            logger.warning(f"Beam decode hatası, greedy fallback: {e}")
            return self._greedy_decode(logits)

    def _greedy_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """Greedy CTC decode (fallback)."""
        probs = self._softmax(logits)
        tokens = np.argmax(probs, axis=-1)

        # CTC collapse
        cleaned = []
        blank_idx = 0
        for t in tokens:
            if t != blank_idx and (not cleaned or t != cleaned[-1]):
                cleaned.append(int(t))

        text = "".join(
            self.vocab[i] if i < len(self.vocab) else " "
            for i in cleaned
        )
        conf = float(np.mean(np.max(probs, axis=-1)))
        return text.strip(), max(min(conf, 1.0), 0.0)

    # ──────────── FACE CUE FUSION ────────────

    def _apply_face_cues(self, text: str, cue) -> Tuple[str, str]:
        """Face cue sinyallerini metne uygula.

        Öncelik sırası: ? > ! > . > ,
        """
        punct = ""

        if cue.suggest_question and not text.endswith("?"):
            text = text.rstrip(".,!") + "?"
            punct = "?"
        elif cue.suggest_exclamation and not text.endswith("!"):
            text = text.rstrip(".,?") + "!"
            punct = "!"
        elif cue.suggest_period and not text.endswith("."):
            text = text.rstrip(",") + "."
            punct = "."
        elif cue.suggest_comma and not text.endswith(","):
            text = text + ","
            punct = ","

        return text, punct

    # ──────────── SUBTITLE BUFFER ────────────

    def _update_subtitle_buffer(self, text: str):
        """Altyazı tamponunu güncelle."""
        # Cümle sonu tespit et
        if text and text[-1] in ".?!":
            self._subtitle_buffer.append(text)
            if len(self._subtitle_buffer) > 10:
                self._subtitle_buffer = self._subtitle_buffer[-10:]
        elif self._subtitle_buffer:
            # Devam eden cümle — son girişi güncelle
            self._subtitle_buffer[-1] = text
        else:
            self._subtitle_buffer.append(text)

    # ──────────── UTILITY ────────────

    def _prepare_labels(self) -> List[str]:
        """pyctcdecode için label listesi hazırlar."""
        labels = []
        for token in self.vocab:
            if token == "<blank>":
                labels.append("")
            else:
                labels.append(token)
        return labels

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
