import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import api.inference as inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_model()
    yield

app = FastAPI(
    title="NCF Core Engine",
    description="Neural Collaborative Filtering — real-time Amazon product recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="Encoded user ID (0-indexed)")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")

class RecommendedItem(BaseModel):
    item_id: int
    score: float

class PredictResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendedItem]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        results = inference.predict(req.user_id, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PredictResponse(
        user_id=req.user_id,
        recommendations=[RecommendedItem(**r) for r in results],
    )