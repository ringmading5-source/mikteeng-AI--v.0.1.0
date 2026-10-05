# Mikteeng AI experimental chatbot

Deploy on Render using New > Blueprint and this repository's render.yaml. For an existing Python Web Service use root directory deploy/mikteeng-chatbot, build command `pip install -r requirements-chatbot-tested.txt && pip install . && python build_model.py`, start command `python start_chatbot.py --host 0.0.0.0`, health path /health, and PYTHON_VERSION=3.12.8.

The October 5 update connects the latest Mikteeng implementation to the chatbot. Existing definitions and sentence continuation are joined by experimental character completion, structured toy-domain planning, and bounded WAV continuation. The build uses repository-owned composition/acoustic seeds and reviewed curriculum examples. No external AI API, training at startup, automatic learning from user messages, or external action execution.

See [chatbot instructions](deploy/mikteeng-chatbot/README.md) for formats, limits, and local tests. After deployment, GET /health must show release `2026-10-05-learning`. This repository has no verified live Mikteeng service URL; a GitHub push alone does not prove Render deployed it. An existing service with auto-deploy enabled may rebuild from main; otherwise deploy the latest commit manually in Render. Do not use the separate Bikting engine service as proof of Mikteeng deployment.
