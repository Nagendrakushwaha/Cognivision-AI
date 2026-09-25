import io
import pyarrow.parquet as pq
from PIL import Image
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from ..services.explain_service import explain_service
from ..services.dataset_service import dataset_service

router = APIRouter(prefix="/api/explain", tags=["Explainability"])

@router.post("/image")
async def explain_uploaded_image(
    file: UploadFile = File(...),
    target_class_idx: Optional[int] = Form(None)
):
    """Generate Grad-CAM visual explanations and regional importance scores for an uploaded receipt."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    res = explain_service.explain_image(image, target_class_idx=target_class_idx)
    if "error" in res and not res.get("is_trained", False):
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@router.post("/sample")
def explain_dataset_sample(
    split: str = Query("test", enum=["train", "test"]),
    idx: int = Query(0, ge=0),
    target_class_idx: Optional[int] = Query(None)
):
    """Generate Grad-CAM visual explanations for a receipt sample from the actual dataset."""
    parquet_path = dataset_service.train_parquet if split == "train" else dataset_service.test_parquet
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if idx >= len(df):
            raise HTTPException(status_code=404, detail=f"Index {idx} out of range for split {split}")
        img_bytes = df.iloc[idx]["image"]["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample image: {str(e)}")

    res = explain_service.explain_image(image, target_class_idx=target_class_idx)
    if "error" in res and not res.get("is_trained", False):
        raise HTTPException(status_code=400, detail=res["error"])
        
    res["sample_key"] = df.iloc[idx]["key"]
    return res
