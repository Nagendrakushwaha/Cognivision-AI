import React from 'react';
import {
  Info,
  Database,
  Cpu,
  Award,
  Sparkles,
  Layers,
  Code2,
  CheckCircle2,
  Terminal,
  FileText
} from 'lucide-react';

export default function AboutProject() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', padding: '32px', maxWidth: '1100px' }}>
      <div>
        <div className="badge badge-purple" style={{ marginBottom: '8px' }}>
          Technical Architecture & Specifications
        </div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, margin: '0 0 8px 0' }}>About COGNIVISION AI</h2>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
          An enterprise-grade Computer Vision and Multimodal Document AI platform built on the ICDAR SROIE
          (Scanned Receipts OCR and Information Extraction) benchmark, featuring local CPU execution, real-time
          model evaluation, Grad-CAM interpretability, and interactive document exploration.
        </p>
      </div>

      {/* Architecture Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Database size={22} color="#8B5CF6" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Authentic Benchmark Dataset</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
            Utilizes 100% authentic scanned receipt data stored across two Parquet partitions:
            <br />• <strong>train-00000-of-00001.parquet</strong> (626 samples, 318 MB)
            <br />• <strong>test-00000-of-00001.parquet</strong> (361 samples, 191 MB)
            <br />Total: <strong>987 receipts</strong> with raw JPEG binaries, OCR tokens, and ground truth entities.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Cpu size={22} color="#06B6D4" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Laptop CPU-Optimized Inference</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
            Architected specifically for local execution on Windows 11 with <strong>AMD Ryzen 5 5500U</strong> (6 cores / 12 threads)
            and 16GB RAM without requiring NVIDIA CUDA or cloud APIs. Features preprocessed Float32 tensor caching for ~15ms inference latency.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Sparkles size={22} color="#10B981" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Explainable AI (Grad-CAM)</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
            Gradient-weighted Class Activation Mapping computed directly on convolutional backbone feature maps.
            Produces blended JET heatmap overlays and spatial quadrant analysis (Header, Body, Footer) to audit model decision-making.
          </p>
        </div>
      </div>

      {/* Machine Learning Pipeline Diagram */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px' }}>
          End-to-End AI Engineering Pipeline
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '12px',
          textAlign: 'center'
        }}>
          {[
            { step: '1. Ingestion', desc: 'Parquet Row Group Parsing' },
            { step: '2. Validation', desc: 'Schema & Null Profiling' },
            { step: '3. Preprocess', desc: 'RGB Normalization (224x224)' },
            { step: '4. Caching', desc: 'Float32 Tensor Cache (.pt)' },
            { step: '5. Training', desc: 'MobileNetV3 / CogniNet' },
            { step: '6. Evaluation', desc: 'Independent Test Benchmark' },
            { step: '7. Grad-CAM', desc: 'Convolutional Heatmaps' },
            { step: '8. Serving', desc: 'FastAPI + React JSX' }
          ].map((item, idx) => (
            <div key={idx} style={{
              padding: '14px 10px',
              borderRadius: '8px',
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)'
            }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#A78BFA', marginBottom: '4px' }}>{item.step}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tech Stack Details */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Technology Stack</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div>
            <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#67E8F9', marginBottom: '8px' }}>Backend & Modeling</h4>
            <ul style={{ listStyleType: 'none', padding: 0, margin: 0, fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <li>• <strong>Python 3.13</strong> (Native Windows x64)</li>
              <li>• <strong>PyTorch 2.12 & Torchvision 0.27</strong> (CPU-optimized build)</li>
              <li>• <strong>FastAPI 0.115 & Uvicorn 0.38</strong> (Asynchronous REST API)</li>
              <li>• <strong>PyArrow & Pandas</strong> (Parquet binary decompression)</li>
              <li>• <strong>Scikit-Learn 1.8</strong> (Confusion matrix, ROC-AUC, classification report)</li>
              <li>• <strong>Pillow & Matplotlib</strong> (Image transforms & Grad-CAM colormaps)</li>
            </ul>
          </div>

          <div>
            <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#A78BFA', marginBottom: '8px' }}>Frontend & Visualization</h4>
            <ul style={{ listStyleType: 'none', padding: 0, margin: 0, fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <li>• <strong>React 19 & Vite 6</strong> (Vanilla JavaScript .jsx)</li>
              <li>• <strong>Recharts</strong> (Interactive SVG graphs & curves)</li>
              <li>• <strong>Lucide React</strong> (Modern UI iconography)</li>
              <li>• <strong>Custom Glassmorphic CSS</strong> (High visual excellence, dark theme)</li>
              <li>• <strong>FastAPI Proxy Integration</strong> (Seamless local routing)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* CLI Commands Reference */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '14px' }}>Local Execution Guide</h3>
        <div style={{
          backgroundColor: '#05070A',
          padding: '16px',
          borderRadius: '8px',
          fontFamily: 'monospace',
          fontSize: '0.8rem',
          color: '#D1D5DB',
          lineHeight: 1.7
        }}>
          <div># 1. Run Model Training from Terminal:</div>
          <div style={{ color: '#10B981' }}>py -3.13 train.py --epochs 5 --model mobilenet_v3 --batch_size 32</div>
          <div style={{ marginTop: '8px' }}># 2. Run Test Set Evaluation:</div>
          <div style={{ color: '#10B981' }}>py -3.13 evaluate.py --model_path models/best_model.pt</div>
          <div style={{ marginTop: '8px' }}># 3. Start FastAPI Backend:</div>
          <div style={{ color: '#10B981' }}>py -3.13 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload</div>
          <div style={{ marginTop: '8px' }}># 4. Start React Frontend Dev Server:</div>
          <div style={{ color: '#10B981' }}>cd frontend && npm run dev</div>
        </div>
      </div>
    </div>
  );
}
