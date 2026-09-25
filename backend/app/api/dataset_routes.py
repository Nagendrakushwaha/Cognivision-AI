from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from ..services.dataset_service import dataset_service

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

@router.get("/summary")
def get_dataset_summary():
    """Returns dataset summary, schema, missing values, class distributions, and image dimensions."""
    return dataset_service.get_summary()

@router.get("/records")
def get_dataset_records(
    split: str = Query("train", enum=["train", "test"]),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None)
):
    """Returns paginated records from train or test set."""
    return dataset_service.get_records(split=split, page=page, page_size=page_size, category=category)

@router.get("/receipt/{split}/{idx}")
def get_receipt_detail(split: str, idx: int):
    """Returns receipt detail with bounding boxes, OCR words, extracted entities, and thumbnail preview."""
    res = dataset_service.get_receipt_detail(split=split, idx=idx)
    if not res:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return res

@router.get("/classes")
def get_classes():
    """Returns available document categories."""
    summary = dataset_service.get_summary()
    return {"classes": summary["classes"]}
