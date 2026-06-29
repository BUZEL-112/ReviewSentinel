# Error Handling

When interacting with the API, you may encounter HTTP error status codes. Here is what they mean and how to resolve them.

## `422 Unprocessable Entity`
**Why:** Your JSON request body is malformed or missing required fields. FastAPI automatically validates the request schema.
**What to do:** Check the schema in [API Endpoints](endpoints.md).
**Example:**
```json
{
  "detail": [
    {
      "loc": ["body", "title"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
*Resolution:* Ensure you are providing the `title` field in your JSON body.

## `503 Service Unavailable`
**Why:** The API process is running, but the underlying machine learning model is not currently loaded into memory. This can happen during initial startup or if the model files are missing from the `artifacts/` directory.
**What to do:** Check the API logs to see why the `InferencePipeline` failed to initialize. If running locally for the first time, ensure you have executed the training pipeline to generate the model.

## `500 Internal Server Error`
**Why:** An unexpected exception occurred during the model forward pass or response serialization.
**What to do:** Check the API container logs for a Python traceback. This usually indicates a bug in the code rather than a client error.
