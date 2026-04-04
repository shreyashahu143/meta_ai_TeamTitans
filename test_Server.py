import requests

BASE = "http://localhost:7860"

# Test 1
print("Testing /health...")
r = requests.get(f"{BASE}/health")
print(f"✓ Health: {r.json()}")

# Test 2
print("\nTesting /reset...")
r = requests.post(f"{BASE}/reset")
data = r.json()
print(f"✓ Reset successful. First email from: {data['sender']}")

# Test 3
print("\nTesting /step (RESPOND)...")
r = requests.post(f"{BASE}/step", json={"action": 1})
result = r.json()
print(f"✓ Step successful. Reward: {result['reward']}")

print("\n✅ All tests passed!")