import io
import pyarrow.parquet as pq
from PIL import Image
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from ..services.predict_service import predict_service
from ..services.dataset_service import dataset_service

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

@router.post("/image")
async def predict_single_image(
    file: UploadFile = File(...),
    include_gradcam: bool = Form(True)
):
    """Run model inference on an uploaded receipt image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    res = predict_service.predict_image(image, include_gradcam=include_gradcam)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@router.post("/sample")
def predict_dataset_sample(
    split: str = Query("test", enum=["train", "test"]),
    idx: int = Query(0, ge=0)
):
    """Run model inference on a receipt from the actual dataset."""
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

    res = predict_service.predict_image(image, include_gradcam=True)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
        
    actual_category = df.iloc[idx]["entities"].get("company", "") if df.iloc[idx]["entities"] else ""
    res["sample_info"] = {
        "key": df.iloc[idx]["key"],
        "idx": idx,
        "split": split,
        "company": actual_category
    }
    return res

@router.post("/batch")
async def predict_batch_images(
    files: List[UploadFile] = File(...)
):
    """Run batch inference on multiple uploaded receipt images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images_list = []
    for file in files:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            images_list.append((file.filename, img))
        except Exception:
            continue

    if not images_list:
        raise HTTPException(status_code=400, detail="Could not decode any valid images from upload")

    res = predict_service.predict_batch(images_list)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@router.get("/batch/sample_test")
def predict_batch_test_samples(count: int = Query(10, ge=2, le=50)):
    """Run batch inference on a batch of test set samples."""
    try:
        table = pq.read_table(dataset_service.test_parquet)
        df = table.to_pandas().head(count)
        images_list = []
        for idx, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images_list.append((f"{row['key']}.jpg", img))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load test samples: {e}")

    res = predict_service.predict_batch(images_list)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res
