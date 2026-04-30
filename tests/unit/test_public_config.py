from app.main import _normalize_clerk_frontend_api


def test_normalize_clerk_frontend_api_accepts_full_url():
    assert (
        _normalize_clerk_frontend_api("https://funny-ocelot-16.clerk.accounts.dev")
        == "funny-ocelot-16.clerk.accounts.dev"
    )


def test_normalize_clerk_frontend_api_accepts_host():
    assert (
        _normalize_clerk_frontend_api("funny-ocelot-16.clerk.accounts.dev")
        == "funny-ocelot-16.clerk.accounts.dev"
    )
