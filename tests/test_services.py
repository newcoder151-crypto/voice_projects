import json
import threading

import numpy as np

from conftest import FakeTranscriber
from voice.dial.service import DialService
from voice.files import ProcessedState, list_audio_files, watch_directory
from voice.models import Utterance
from voice.transcribe.service import TranscribeService


def test_dial_file_sends_udp(make_cfg, receiver, tmp_path):
    sock, host, port = receiver
    cfg = make_cfg({"dial.udp.host": host, "dial.udp.port": port, "dial.udp.payload_template": "{number}"})
    transcriber = FakeTranscriber()
    service = DialService(cfg, transcriber)
    sent = service.process_file(tmp_path / "voice.wav")
    assert sent == ["9876543210"]
    assert sock.recvfrom(1024)[0] == b"9876543210"
    assert transcriber.calls[0][2] == "en"  # dial.language is applied
    service.close()


def test_dial_live_phrase_and_duplicate_suppression(make_cfg, receiver):
    sock, host, port = receiver
    cfg = make_cfg({"dial.udp.host": host, "dial.udp.port": port})
    service = DialService(cfg, FakeTranscriber("dial 5 5 5 0 1 0 0"))
    utt = Utterance(np.zeros(16000, dtype=np.float32), 3.0, 4.0)
    assert service.handle_utterance(utt) == ["5550100"]
    assert service.handle_utterance(utt) == []  # duplicate within dedupe window
    assert sock.recvfrom(1024)[0] == b"5550100"
    service.close()


def test_dial_ignores_speech_without_command(make_cfg, receiver):
    _, host, port = receiver
    cfg = make_cfg({"dial.udp.host": host, "dial.udp.port": port})
    service = DialService(cfg, FakeTranscriber("what a lovely day"))
    assert service.handle_utterance(Utterance(np.zeros(8000, dtype=np.float32), 0, 0.5)) == []
    service.close()


def test_transcribe_file_writes_txt_and_json(make_cfg, tmp_path):
    cfg = make_cfg()
    service = TranscribeService(cfg, FakeTranscriber("hello world"))
    txt, js = service.process_file(tmp_path / "meeting.mp3")
    assert txt.read_text().strip() == "[00:00:00.000 --> 00:00:02.500] hello world"
    doc = json.loads(js.read_text())
    assert doc["source"] == "meeting.mp3" and doc["text"] == "hello world"
    assert doc["segments"][0]["end"] == 2.5 and doc["language"] == "en"


def test_transcribe_live_session_appends(make_cfg):
    cfg = make_cfg({"transcribe.timestamps_in_txt": False})
    service = TranscribeService(cfg, FakeTranscriber("good morning"))
    service.handle_utterance(Utterance(np.zeros(16000, dtype=np.float32), 10.0, 11.0))
    service.handle_utterance(Utterance(np.zeros(16000, dtype=np.float32), 20.0, 21.0))
    session = service._session
    service.close()
    assert session.txt_path.read_text() == "good morning\ngood morning\n"
    lines = [json.loads(l) for l in session.jsonl_path.read_text().splitlines()]
    assert [l["start"] for l in lines] == [10.0, 20.0]


def test_watcher_processes_once_and_retries_failures(tmp_path):
    import os, time
    inbox = tmp_path / "in"
    inbox.mkdir()
    good, bad = inbox / "a.wav", inbox / "b.mp3"
    for f in (good, bad, inbox / "notes.txt"):
        f.write_bytes(b"x")
        os.utime(f, (time.time() - 60,) * 2)
    state = ProcessedState(tmp_path / "state" / "p.json")
    stop = threading.Event()
    calls = []

    def handler(path):
        calls.append(path.name)
        if path.name == "b.mp3":
            raise RuntimeError("bad audio")
        if calls.count("a.wav") >= 1 and calls.count("b.mp3") >= 2:
            stop.set()

    def run():
        watch_directory(inbox, [".wav", ".mp3"], state, handler, 0.01, 0, 2, stop)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(2)
    assert calls.count("a.wav") == 1          # done once, never repeated
    assert calls.count("b.mp3") == 2          # retried up to max_retries then skipped
    assert "notes.txt" not in calls
    assert ProcessedState(tmp_path / "state" / "p.json").is_done(good)
    assert [p.name for p in list_audio_files(inbox, [".wav"])] == ["a.wav"]
