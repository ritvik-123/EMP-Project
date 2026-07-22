from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

response = client.post(
    "/predict",
    json={
        "sentence": "I started believing I was less capable because of how people treated me.",
        "use_llm_backup": False
    }
)

print("Status code:", response.status_code)
print(response.json())
