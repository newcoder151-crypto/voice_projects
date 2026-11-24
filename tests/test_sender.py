from voice.dial.sender import UdpSender


def test_sends_payload(receiver):
    sock, host, port = receiver
    sender = UdpSender(host, port, payload_template="DIAL:{number}")
    assert sender.send("9876543210") is True
    data, _ = sock.recvfrom(1024)
    assert data == b"DIAL:9876543210"
    sender.close()


def test_dedupe_window(receiver):
    _, host, port = receiver
    now = [100.0]
    sender = UdpSender(host, port, dedupe_seconds=5, clock=lambda: now[0])
    assert sender.send("123") is True
    now[0] = 102.0
    assert sender.send("123") is False
    assert sender.send("456") is True
    now[0] = 106.0
    assert sender.send("123") is True
    sender.close()


def test_dry_run_sends_nothing(receiver):
    sock, host, port = receiver
    sock.settimeout(0.2)
    sender = UdpSender(host, port, dry_run=True)
    assert sender.send("123") is True
    try:
        sock.recvfrom(1024)
        raise AssertionError("a datagram was sent during a dry run")
    except OSError:
        pass
