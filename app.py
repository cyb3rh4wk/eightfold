from fastapi import FastAPI, HTTPException
from models import Query, Response
from persistence import Persistence
from workflow import graph
import logging

# initialize FastAPI app
app = FastAPI(title="AI Support Agent")

# initialize state manager
db = Persistence()

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/conversation", response_model=Response)
async def handle_query(query: Query):
    """Handles incoming customer queries."""
    session_id = query.session_id
    text = query.text

    # load session state
    state = db.load_state(session_id)
    print('[DEBUG] Loaded State: ', state)
    state["query"] = text

    try:
        # run the LangGraph workflow
        final_state = await graph.ainvoke(state)
        
        # save the updated state
        db.save_state(session_id, final_state)
        
        logger.info(f"Processed query for session {session_id}: {text}")
        return Response(response=final_state["response"])
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
