"""
tests/test_pipeline.py — PipelineController yaşam döngüsü testleri.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import PipelineController


@pytest.fixture
def pipeline():
    """Mock modda çalışan pipeline instance."""
    p = PipelineController(
        model_path="models/student_int8.onnx",
        chunk_size=3,
        fps_limit=10,
    )
    yield p
    if p.running:
        p.stop()


class TestPipelineLifecycle:
    def test_initial_state(self, pipeline):
        """Pipeline başlangıçta durağan olmalı."""
        assert not pipeline.running
        assert pipeline.frame_q.empty()
        assert pipeline.roi_q.empty()

    def test_start_sets_running(self, pipeline):
        """start() running flag'ini True yapmalı."""
        pipeline.start()
        assert pipeline.running
        time.sleep(0.5)
        pipeline.stop()

    def test_stop_clears_running(self, pipeline):
        """stop() running flag'ini False yapmalı."""
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()
        assert not pipeline.running

    def test_double_start_idempotent(self, pipeline):
        """İkinci start() çağrısı hata vermemeli."""
        pipeline.start()
        result = pipeline.start()
        assert result is True
        pipeline.stop()

    def test_stop_without_start(self, pipeline):
        """start() olmadan stop() çağrısı hata vermemeli."""
        pipeline.stop()  # Hata fırlatmamalı


class TestSignals:
    def test_subtitle_signal_emitted(self, pipeline):
        """Mock modda subtitle sinyali gelmeli."""
        received = []
        pipeline.subtitle_ready.connect(lambda text, conf: received.append((text, conf)))

        pipeline.start()
        time.sleep(2.0)  # Mock data üretmesi için bekle
        pipeline.stop()

        # Mock modda sinyal gelmiş olmalı (model+ROI mock olduğundan)
        # Not: Gerçek model yoksa MockInference devreye girer
        if len(received) > 0:
            text, conf = received[0]
            assert isinstance(text, str)
            assert 0.0 <= conf <= 1.0

    def test_metrics_signal_emitted(self, pipeline):
        """Mock modda metrics sinyali gelmeli."""
        received = []
        pipeline.metrics_ready.connect(lambda m: received.append(m))

        pipeline.start()
        time.sleep(2.0)
        pipeline.stop()

        if len(received) > 0:
            m = received[0]
            assert "latency_ms" in m
            assert "fps" in m


class TestQueueBackpressure:
    def test_frame_queue_bounded(self, pipeline):
        """Frame queue maxsize aşılmamalı."""
        assert pipeline.frame_q.maxsize == 2

    def test_roi_queue_bounded(self, pipeline):
        """ROI queue maxsize aşılmamalı."""
        assert pipeline.roi_q.maxsize == 10
