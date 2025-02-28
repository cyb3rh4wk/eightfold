# How It Works

## 1. Query Submission
- **POST Request:** The client sends a POST request to `/query` with a JSON payload that includes:
  - `session_id`
  - `text`

  **Example Payload:**
  ```json
  {"session_id": "user123", "text": "My laptop won’t turn on"}
  ```
- **Validation:** The `Query` model in `models.py` validates the input.

## 2. State Management
- **Session Handling:** `persistence.py` either loads the current session state or initializes a new one.
- **History Update:** The submitted query is appended to the session’s history.

## 3. Workflow Execution
The `workflow.py` module executes the LangGraph graph through the following steps:
- **Query Ingestion:** Adds the query to the state.
- **NLP Processing:** Invokes `groq_client.py` to extract structured information (e.g., issue, priority).
- **Decision Making:** Determines whether to auto-respond or escalate the query based on its priority/urgency.
- **Response Generation:** Constructs a response based on the decision made.

## 4. Response
- **State Saving:** The updated state is saved.
- **Client Response:** The final response is returned to the client as a `Response` object.

---

# Running the Application

## Step 1: Install Dependencies
Run the following command in your terminal:
```bash
pip install fastapi uvicorn groq langgraph pydantic
```

## Step 2: Set the API Key
- Open `config.py`
- Replace `"your_groq_api_key"` with your actual Groq API key.

## Step 3: Run the Application
Start the app by executing:
```bash
python app.py
```

## Step 4: Test with a Query
Send a test POST request using `curl`:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "user123", "text": "My laptop won’t turn on after I updated it yesterday."}'
```

## Expected Response
The expected JSON response should appear as follows:
```json
{
  "response": "We have identified your issue as 'won’t turn on'. Please try this solution: [Solution Placeholder]"
}
```
