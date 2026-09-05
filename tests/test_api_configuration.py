import pytest

from src.api import _parse_allowed_origins


def test_cors_origins_use_safe_local_defaults() -> None:
    assert _parse_allowed_origins(None) == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_cors_origins_accept_explicit_https_hosts() -> None:
    assert _parse_allowed_origins(
        "https://radar.example, https://admin.example/"
    ) == [
        "https://radar.example",
        "https://admin.example",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "https://user:password@example.com",
        "https://example.com/path",
        "https://example.com:not-a-port",
        "https://:443",
        "javascript:alert(1)",
        ", ,",
    ],
)
def test_cors_origins_reject_unsafe_values(
    value: str,
) -> None:
    with pytest.raises(RuntimeError):
        _parse_allowed_origins(value)
