## FastAPI Fundamentals

**Task:**

1. Create a **FastAPI application** with the following endpoints:

   * `GET /`
     Returns a short description of the API.
   * `GET /health`
     Returns a simple status message (e.g., `"status": "ok"`).
   * `POST /echo`
     Accepts JSON input and returns the same data back in the response.
2. Use **Pydantic models** to define and validate request data for the `POST` endpoint.
3. Run the application and test all endpoints using **Swagger UI** (`/docs`).