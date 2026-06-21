import httpx, time

base = "http://localhost:8000"
client = httpx.Client()

print("=== Logging normal inferences ===")
for age, income in [(25,40000),(30,55000),(35,70000),(40,85000),(45,100000)]:
    r = client.post(f"{base}/log", json={
        "features": {"age": age, "income": income, "score": 0.5+age/100},
        "prediction": 1 if income > 60000 else 0,
        "label": 1 if income > 60000 else 0,
    })
    rid = r.json()["request_id"]
    print(f"  logged age={age} income={income}: {r.json()}")

print("\n=== Setting reference baseline ===")
r = client.post(f"{base}/reference/baseline")
print(f"  {r.json()}")

print("\n=== Logging shifted inferences (should trigger drift) ===")
for age, income in [(70,40000),(80,55000),(90,70000),(100,85000),(110,100000)]:
    r = client.post(f"{base}/log", json={
        "features": {"age": age, "income": income, "score": 0.5+age/100},
        "prediction": 1 if income > 60000 else 0,
    })
    print(f"  logged age={age}: {r.json()}")

time.sleep(1)

print("\n=== Latest drift results ===")
r = client.get(f"{base}/drift/latest")
for res in r.json():
    sig = "⚠️ DRIFT" if res["significant"] else "✅ ok"
    print(f"  {res['feature_name']}: p={res['p_value']:.4f} adj-p={res['corrected_p_value']:.4f} {sig}")

print("\n=== Health check ===")
r = client.get(f"{base}/health")
print(f"  {r.json()}")
