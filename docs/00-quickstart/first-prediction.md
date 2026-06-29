# First Prediction

Want to see ReviewSentinel in action? You can have the API up and running in a few minutes.

## Start the Services

```bash
docker-compose -f docker/docker-compose.yaml up --build -d
```

## Run a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Great purchase", "text": "This exceeded my expectations."}'
```

**Success:** You should see `"label": "positive"` in the JSON response within 30 seconds.
