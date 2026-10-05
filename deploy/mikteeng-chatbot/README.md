# Existing deployment compatibility

This directory contains only a launcher and installation configuration. The old chatbot, learners and datasets have been removed. Existing Render services using this directory can retain their old build/start commands: requirements install the root RSPM package, `build_model.py` validates its checkpoint, and `start_chatbot.py` launches its vector inference service.

New deployments should use the root `render.yaml`. There is only one active model implementation: `src/mikteeng_rspm`.
