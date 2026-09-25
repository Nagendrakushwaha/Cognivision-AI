import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api.dataset_routes import router as dataset_router
from .api.model_routes import router as model_router
from .api.predict_routes import router as predict_router
from .api.explain_routes import router as explain_router

app = FastAPI(
    title="COGNIVISION AI",
    description="Enterprise-Grade Computer Vision & Multimodal Receipt Document Intelligence Platform",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(dataset_router)
app.include_router(model_router)
app.include_router(predict_router)
app.include_router(explain_router)

@app.get("/")
def root():
    return {
        "project": "COGNIVISION AI",
        "description": "Multimodal Computer Vision & Document Intelligence System",
        "dataset": "SROIE Benchmark (Scanned Receipts OCR and Information Extraction)",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/api/health")
def health_check():
    train_cache_exists = os.path.exists("artifacts/cache/train_cache.pt")
    test_cache_exists = os.path.exists("artifacts/cache/test_cache.pt")
    model_exists = os.path.exists("models/best_model.pt") or os.path.exists("models/latest_model.pt")
    
    return {
        "status": "healthy",
        "cache_ready": train_cache_exists and test_cache_exists,
        "model_trained": model_exists,
        "device": "CPU (AMD Ryzen 5 5500U optimized)"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": str(request.url)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
