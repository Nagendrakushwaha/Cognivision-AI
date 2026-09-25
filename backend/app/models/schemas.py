from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DatasetSummary(BaseModel):
    total_records: int
    train_records: int
    test_records: int
    num_features: int
    target_column: str
    num_classes: int
    classes: List[str]
    class_distribution_train: Dict[str, int]
    class_distribution_test: Dict[str, int]
    missing_values: Dict[str, int]
    image_dimension_stats: Dict[str, Any]
    ocr_words_stats: Dict[str, Any]

class SampleRecord(BaseModel):
    idx: int
    key: str
    category: str
    label: int
    split: str
    width: int
    height: int
    num_words: int
    company: str
    date: str
    address: str
    total: str

class TrainConfig(BaseModel):
    model_name: str = Field(default="mobilenet_v3", description="mobilenet_v3 or cogninet_cnn")
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=32, ge=8, le=128)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    optimizer: str = Field(default="Adam", description="Adam, AdamW, or SGD")
    loss_function: str = Field(default="CrossEntropyLoss")
    validation_split: float = Field(default=0.2, ge=0.05, le=0.4)
    random_seed: int = Field(default=42)
    freeze_backbone: bool = Field(default=True)

class TrainEpochLog(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    duration_seconds: float

class TrainStatus(BaseModel):
    status: str # "idle", "training", "completed", "error"
    current_epoch: int
    total_epochs: int
    progress_percent: float
    current_train_loss: Optional[float] = None
    current_val_loss: Optional[float] = None
    current_train_acc: Optional[float] = None
    current_val_acc: Optional[float] = None
    history: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class ClassMetric(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int

class EvaluationReport(BaseModel):
    is_trained: bool
    model_name: str
    evaluated_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    roc_auc: Optional[float] = None
    confusion_matrix: List[List[int]]
    classes: List[str]
    class_wise_metrics: List[ClassMetric]
    evaluated_at: str

class TopKPrediction(BaseModel):
    class_name: str
    probability: float
    percentage: float

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    confidence_percentage: float
    top_k: List[TopKPrediction]
    image_preview_base64: Optional[str] = None
    gradcam_base64: Optional[str] = None
    inference_time_ms: float
    model_used: str

class BatchPredictionItem(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    confidence_percentage: float
    top_2_class: str
    top_2_confidence: float

class BatchPredictionResponse(BaseModel):
    total_processed: int
    results: List[BatchPredictionItem]
    model_used: str
    inference_time_ms: float

class MultimodalEntity(BaseModel):
    text: str
    label: str # "company", "date", "address", "total", "other"
    bbox: List[int] # [x1, y1, x2, y2]

class MultimodalDocumentResponse(BaseModel):
    key: str
    category: str
    image_dimensions: List[int]
    entities_extracted: Dict[str, str]
    words_count: int
    tokens: List[MultimodalEntity]
    image_base64: Optional[str] = None
