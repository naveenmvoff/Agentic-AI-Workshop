from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
from agents import run_agentic_editor, get_session_info, reset_session  # ✅ make sure agents.py has these

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(
    title="Agentic AI Website Editor",
    description="AI-powered website editor using agentic architecture",
    version="1.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response models
class EditCommand(BaseModel):
    command: str

class ApiResponse(BaseModel):
    status: str
    message: str = ""
    data: Dict[str, Any] = {}

# Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Agentic AI Website Editor is running"}

@app.post("/process_command")
async def process_command(data: EditCommand) -> ApiResponse:
    try:
        if not data.command.strip():
            raise HTTPException(status_code=400, detail="Command cannot be empty")

        logger.info(f"Processing command: {data.command}")
        result = run_agentic_editor(data.command.strip())
        return ApiResponse(
            status="success",
            message="Command processed successfully",
            data=result
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process command: {str(e)}")

@app.get("/session/info")
async def session_info():
    return get_session_info()

@app.post("/session/reset")
async def session_reset():
    reset_session()
    return {"message": "Session reset successful"}

# Server run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
