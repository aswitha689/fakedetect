# Explainable AI Fake News Detection
1. Install dependencies:
   pip install -r requirements.txt
2. Train ML pipeline:
   python backend/train.py
3. (Optional) Set Fact Check API key:
   export FACTCHECK_API_KEY="your_key"  # Windows: $env:FACTCHECK_API_KEY="your_key"
4. Start server:
   uvicorn backend.app:app --port 8000
5. Open UI: http://127.0.0.1:8000
