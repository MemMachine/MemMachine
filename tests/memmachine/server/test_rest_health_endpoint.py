from fastapi.testclient import TestClient

from memmachine.server.app import app

# Create a test client for the FastAPI app.
# Setting raise_server_exceptions=False ensures that server-side exceptions
# are returned as HTTP 500 responses instead of being raised in the test code.
client = TestClient(app, raise_server_exceptions=False)


def test_health_check_propagates_unexpected_exceptions(monkeypatch):
    """
    This test verifies two behaviors of the /health endpoint:
    1. When expected errors occur (e.g., missing memory managers), the endpoint returns a 503 status code and a useful error message.
    2. When an unexpected exception occurs, the endpoint returns a generic 500 Internal Server Error response.
    """

    # Monkeypatch global memory managers to None to simulate an expected error condition.
    import memmachine.server.app as app_module

    monkeypatch.setattr(app_module, "profile_memory", None)
    monkeypatch.setattr(app_module, "episodic_memory", None)

    # Call the /health endpoint and verify it returns a 503 status code
    # and a specific error message about memory managers not being initialized.
    response = client.get("/health")
    assert response.status_code == 503
    assert "Memory managers not initialized" in response.text

    # Monkeypatch the /health endpoint itself to raise an unexpected exception.
    # This simulates a programming error or bug in the health check logic.
    async def broken_health_check():
        raise ValueError("Unexpected error")

    # Remove the original /health route and replace it with the broken one.
    app.router.routes = [
        route for route in app.router.routes if route.path != "/health"
    ]
    app.get("/health")(broken_health_check)

    # Call the /health endpoint again and verify it returns a 500 status code
    # and a generic error message (FastAPI does not expose exception details by default).
    response = client.get("/health")
    assert response.status_code == 500
    assert "Internal Server Error" in response.text


def test_health_check_returns_healthy_status(monkeypatch):
    """
    This test verifies that the /health endpoint returns a healthy status and useful information
    when the application is properly initialized.
    """

    # Monkeypatch global memory managers to simulate healthy state
    import memmachine.server.app as app_module

    class DummyMemoryManager:
        pass

    monkeypatch.setattr(app_module, "profile_memory", DummyMemoryManager())
    monkeypatch.setattr(app_module, "episodic_memory", DummyMemoryManager())

    # Remove any monkeypatched /health route and restore the original
    app.router.routes = [
        route for route in app.router.routes if route.path != "/health"
    ]
    app.get("/health")(app_module.health_check)

    # Call the /health endpoint and verify it returns a 200 status code
    # and includes expected health information in the response.
    response = client.get("/health")
    print("Health endpoint response:", response.text)  # Debug output
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "memmachine"
    assert "version" in json_data
    assert json_data["memory_managers"]["profile_memory"] is True
    assert json_data["memory_managers"]["episodic_memory"] is True
