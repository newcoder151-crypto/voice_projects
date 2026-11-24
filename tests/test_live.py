import numpy as np

from voice.live import LiveSegmenter, run_live

FRAME = 512


class EnergyVAD:
    """Test VAD: speech when the frame is loud; 'end' after N quiet frames."""

    def __init__(self, quiet_frames=5):
        self.quiet_frames = quiet_frames
        self.active = False
        self.quiet = 0

    def process(self, frame):
        loud = float(np.abs(frame).mean()) > 0.1
        if not self.active:
            if loud:
                self.active, self.quiet = True, 0
                return "start"
            return None
        self.quiet = 0 if loud else self.quiet + 1
        if self.quiet >= self.quiet_frames:
            self.active = False
            return "end"
        return None

    def reset(self):
        self.active, self.quiet = False, 0


def tone(n_frames):
    return [np.full(FRAME, 0.5, dtype=np.float32) for _ in range(n_frames)]


def silence(n_frames):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n_frames)]


def make(**kw):
    defaults = dict(vad=EnergyVAD(), sample_rate=16000, frame_samples=FRAME,
                    pre_roll_ms=64, min_speech_ms=100, max_utterance_sec=20,
                    trailing_silence_ms=160)
    defaults.update(kw)
    return LiveSegmenter(**defaults)


def feed_all(seg, frames):
    out = []
    for f in frames:
        out += seg.feed(f)
    return out


def test_cuts_two_phrases_with_pre_roll():
    seg = make()
    out = feed_all(seg, silence(10) + tone(20) + silence(10) + tone(15) + silence(10))
    assert len(out) == 2
    first = out[0]
    # 2 pre-roll frames + 20 speech + 5 trailing quiet frames
    assert len(first.audio) == (2 + 20 + 5) * FRAME
    assert abs(first.start - 8 * FRAME / 16000) < 1e-6
    assert out[1].start > first.end


def test_drops_blips_shorter_than_min_speech():
    seg = make(min_speech_ms=300)
    assert feed_all(seg, silence(5) + tone(2) + silence(10)) == []


def test_flushes_unfinished_phrase_at_stream_end():
    seg = make()
    assert feed_all(seg, tone(20)) == []
    tail = seg.flush()
    assert len(tail) == 1 and len(tail[0].audio) == 20 * FRAME


def test_forced_flush_on_long_speech():
    seg = make(max_utterance_sec=1.0)  # 31 frames
    out = feed_all(seg, tone(70))
    assert len(out) >= 2


def test_run_live_handles_phrases_and_survives_handler_errors():
    seg = make()
    frames = silence(5) + tone(20) + silence(10) + tone(20) + silence(10)

    class Source:
        closed = False

        def frames(self):
            yield from frames

        def close(self):
            self.closed = True

    import threading
    seen = []

    def handler(utt):
        seen.append(utt)
        if len(seen) == 1:
            raise RuntimeError("boom")

    src = Source()
    run_live(src, seg, handler, threading.Event())
    assert len(seen) == 2 and src.closed
