# Mikteeng AI experimental chatbot
Deploy the repository using Render New > Blueprint, select this repository and apply render.yaml. Alternatively create a Python Web Service with root directory deploy/mikteeng-chatbot, build command `pip install -r requirements-chatbot-tested.txt && pip install . && python build_model.py`, start command `python start_chatbot.py --host 0.0.0.0`, health path /health, and PYTHON_VERSION=3.12.8.

Build recreates the definition and sentence continuation heads from the existing reviewed training examples. It does not train at server startup. Earlier role/passage heads and the separate arithmetic code experiment are not served by this release. No automatic learning from user messages, no external AI API. Text only; audio/image understanding is not exposed.

GET /health returns readiness; GET /api/coverage lists the curriculum; POST /api/chat accepts message, mode (generated/reference/continuation), and context. Predictions are experimental. No history is stored on the server. Free hosting availability and memory capacity must be checked in the actual deployment logs.
