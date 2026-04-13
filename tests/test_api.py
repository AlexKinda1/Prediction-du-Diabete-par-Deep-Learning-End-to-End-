from fastapi.testclient import TestClient
from deployment.app import app


def test_health():
    client = TestClient(app)
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok'}
