import base64
import hashlib
import hmac

from app.main import _verify_recall_signature


def _signed_headers(secret: str, body: bytes) -> dict[str, str]:
    msg_id = "msg_test"
    timestamp = "1731705121"
    key = base64.b64decode(secret.removeprefix("whsec_"))
    payload = body.decode("utf-8")
    signature = base64.b64encode(
        hmac.new(key, f"{msg_id}.{timestamp}.{payload}".encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "webhook-id": msg_id,
        "webhook-timestamp": timestamp,
        "webhook-signature": f"v1,{signature}",
    }


def test_verify_recall_signature_accepts_valid_workspace_secret_signature():
    secret = "whsec_d2ViaG9va19zZWNyZXRfdGVzdA=="
    body = b'{"event":"bot.transcription"}'

    assert _verify_recall_signature(body, _signed_headers(secret, body), secret) is True


def test_verify_recall_signature_rejects_tampered_body():
    secret = "whsec_d2ViaG9va19zZWNyZXRfdGVzdA=="
    body = b'{"event":"bot.transcription"}'

    assert _verify_recall_signature(
        b'{"event":"tampered"}',
        _signed_headers(secret, body),
        secret,
    ) is False


def test_verify_recall_signature_rejects_missing_headers():
    assert _verify_recall_signature(
        b'{"event":"bot.transcription"}',
        {},
        "whsec_d2ViaG9va19zZWNyZXRfdGVzdA==",
    ) is False
