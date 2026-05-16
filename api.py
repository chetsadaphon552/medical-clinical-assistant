from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import MedicalSymptomAssistant
import uvicorn

app = FastAPI(
    title="Clinical Decision Support API",
    description="API สำหรับวิเคราะห์อาการผู้ป่วยและช่วยวินิจฉัยโรคเบื้องต้น (Medical Agentic RAG)",
    version="1.0.0"
)

# Initialize the agent
agent = MedicalSymptomAssistant()

class SymptomRequest(BaseModel):
    symptoms: str

class SymptomResponse(BaseModel):
    analysis_report: str

@app.post("/api/analyze", response_model=SymptomResponse)
async def analyze_symptoms(request: SymptomRequest):
    try:
        if not request.symptoms.strip():
            raise HTTPException(status_code=400, detail="Symptoms cannot be empty.")
        
        response_text = agent.query(request.symptoms)
        return SymptomResponse(analysis_report=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
