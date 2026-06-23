# Test script to verify video endpoint
import httpx
import sys

# Test 1: Health check
try:
    resp = httpx.get("http://127.0.0.1:8000/api/health", timeout=5)
    print(f"Health check: {resp.status_code} - {resp.text}")
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)

# Test 2: Video endpoint
test_path = r"D:\j\2222.mp4"
url = f"http://127.0.0.1:8000/api/video?file_path={httpx.utils.quote(test_path)}"
print(f"\nRequest URL: {url}")

try:
    resp = httpx.head(url, timeout=5)
    print(f"Video endpoint HEAD: {resp.status_code}")
except httpx.HTTPStatusError as e:
    print(f"Video endpoint error: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"Video endpoint failed: {e}")

# Test 3: List all routes
try:
    resp = httpx.get("http://127.0.0.1:8000/openapi.json", timeout=5)
    data = resp.json()
    paths = list(data.get("paths", {}).keys())
    print(f"\nAvailable API routes: {paths}")
except Exception as e:
    print(f"OpenAPI check failed: {e}")
