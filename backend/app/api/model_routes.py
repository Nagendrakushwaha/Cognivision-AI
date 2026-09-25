from fastapi import APIRouter, HTTPException
from ..models.schemas import TrainConfig
from ..services.train_service import train_service
from ..services.eval_service import eval_service

router = APIRouter(prefix="/api/model", tags=["Model"])

@router.get("/status")
def get_training_status():
    """Returns real-time model training status, progress, and history logs."""
    return train_service.get_status()

@router.post("/train")
def start_model_training(config: TrainConfig):
    """Triggers model training with configurable parameters."""
    res = train_service.start_training(config.dict())
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("message", "Training could not be started"))
    return res

@router.get("/performance")
def get_model_performance():
    """Returns actual test set evaluation metrics, confusion matrix, and class-wise breakdown."""
    return eval_service.get_latest_metrics()

@router.post("/evaluate")
def trigger_evaluation():
    """Manually evaluates saved model on the test dataset."""
    res = eval_service.evaluate_model()
    if not res.get("is_trained", False):
        raise HTTPException(status_code=400, detail=res.get("error", "Evaluation failed"))
    return res
