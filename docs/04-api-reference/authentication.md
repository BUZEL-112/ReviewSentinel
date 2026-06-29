# Authentication & Security

> [!WARNING]
> **ReviewSentinel currently implements NO authentication or authorization on its API endpoints.** 
> 
> The system is designed to be deployed securely behind an internal VPC, an API gateway, or a service mesh that handles authentication before traffic reaches the FastAPI service. **Do not expose the raw ReviewSentinel API directly to the public internet.**

---

## Current Security Posture

- **No API Keys:** All endpoints (`/predict`, `/health`, `/metrics`) accept requests from any client that can route to the service.
- **No Rate Limiting:** The FastAPI application does not enforce request quotas or rate limits. The model runs in a thread pool and will attempt to process all requests, which could lead to resource exhaustion under denial-of-service conditions.
- **Internal Only:** The system assumes it is operating in a trusted network environment.

---

## Production Recommendations

If you intend to deploy ReviewSentinel in a production environment where it might receive untrusted traffic, you must implement authentication at a higher layer in your infrastructure.

### Option 1: API Gateway (Recommended)

Place an API Gateway (e.g., AWS API Gateway, Kong, Apigee) in front of the Kubernetes or Docker Compose deployment.
- Configure the Gateway to require API keys or OAuth2 tokens.
- Configure the Gateway to enforce rate limits (e.g., 100 requests per minute per client).
- Restrict ReviewSentinel's ingress so it only accepts traffic from the Gateway's IP addresses.

### Option 2: Service Mesh

If deploying in a Kubernetes environment using a service mesh (e.g., Istio, Linkerd):
- Enforce mutual TLS (mTLS) between your microservices and ReviewSentinel.
- Use Istio `AuthorizationPolicy` to ensure only specific, authenticated microservices can call the `/predict` endpoints.

### Option 3: FastAPI Middleware (Code Modification required)

If you must implement authentication directly within the ReviewSentinel codebase, you can modify `src/api/api.py` to use FastAPI's built-in `Depends` mechanism.

A minimal API key implementation would look like this:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# This would need to be loaded securely from the environment
EXPECTED_API_KEY = "your-secure-key"

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

# Apply to an endpoint
@app.post("/predict")
async def predict(request: ReviewRequest, api_key: str = Depends(get_api_key)):
    # ... existing implementation ...
```

If implementing this, ensure the `EXPECTED_API_KEY` is injected securely via Kubernetes Secrets or Docker Compose `.env` files, and never committed to version control.
