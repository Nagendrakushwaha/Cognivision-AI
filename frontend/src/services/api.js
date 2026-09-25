const BASE_URL = '/api';

export const api = {
  // Dataset endpoints
  async getDatasetSummary() {
    const res = await fetch(`${BASE_URL}/dataset/summary`);
    if (!res.ok) throw new Error('Failed to fetch dataset summary');
    return res.json();
  },

  async getDatasetRecords(split = 'train', page = 1, pageSize = 20, category = null) {
    let url = `${BASE_URL}/dataset/records?split=${split}&page=${page}&page_size=${pageSize}`;
    if (category && category !== 'All') {
      url += `&category=${encodeURIComponent(category)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch records');
    return res.json();
  },

  async getReceiptDetail(split, idx) {
    const res = await fetch(`${BASE_URL}/dataset/receipt/${split}/${idx}`);
    if (!res.ok) throw new Error('Failed to fetch receipt detail');
    return res.json();
  },

  // Model endpoints
  async getTrainingStatus() {
    const res = await fetch(`${BASE_URL}/model/status`);
    if (!res.ok) throw new Error('Failed to fetch training status');
    return res.json();
  },

  async startTraining(config) {
    const res = await fetch(`${BASE_URL}/model/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to start training' }));
      throw new Error(err.detail || 'Failed to start training');
    }
    return res.json();
  },

  async getModelPerformance() {
    const res = await fetch(`${BASE_URL}/model/performance`);
    if (!res.ok) throw new Error('Failed to fetch model performance');
    return res.json();
  },

  async evaluateModel() {
    const res = await fetch(`${BASE_URL}/model/evaluate`, { method: 'POST' });
    if (!res.ok) throw new Error('Evaluation failed');
    return res.json();
  },

  // Prediction endpoints
  async predictImage(file, includeGradcam = true) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_gradcam', includeGradcam ? 'true' : 'false');

    const res = await fetch(`${BASE_URL}/predict/image`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
      throw new Error(err.detail || 'Prediction failed');
    }
    return res.json();
  },

  async predictSample(split = 'test', idx = 0) {
    const res = await fetch(`${BASE_URL}/predict/sample?split=${split}&idx=${idx}`, {
      method: 'POST'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Sample prediction failed' }));
      throw new Error(err.detail || 'Sample prediction failed');
    }
    return res.json();
  },

  async predictBatch(files) {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const res = await fetch(`${BASE_URL}/predict/batch`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Batch prediction failed' }));
      throw new Error(err.detail || 'Batch prediction failed');
    }
    return res.json();
  },

  async predictBatchTestSamples(count = 10) {
    const res = await fetch(`${BASE_URL}/predict/batch/sample_test?count=${count}`);
    if (!res.ok) throw new Error('Failed to run batch test prediction');
    return res.json();
  },

  // Explainability endpoints
  async explainImage(file, targetClassIdx = null) {
    const formData = new FormData();
    formData.append('file', file);
    if (targetClassIdx !== null) {
      formData.append('target_class_idx', targetClassIdx);
    }

    const res = await fetch(`${BASE_URL}/explain/image`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Explainability generation failed' }));
      throw new Error(err.detail || 'Explainability generation failed');
    }
    return res.json();
  },

  async explainSample(split = 'test', idx = 0, targetClassIdx = null) {
    let url = `${BASE_URL}/explain/sample?split=${split}&idx=${idx}`;
    if (targetClassIdx !== null) {
      url += `&target_class_idx=${targetClassIdx}`;
    }
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Sample explanation failed' }));
      throw new Error(err.detail || 'Sample explanation failed');
    }
    return res.json();
  },

  // Health
  async getHealth() {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  }
};
