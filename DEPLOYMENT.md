# Deploy Mikteeng RSPM

The checked-in Render blueprint is configured for the RSPM inference service. The former `deploy/mikteeng-chatbot` root also contains a thin compatibility launcher: its old build and start commands now install and run the same root RSPM package. No legacy learner is retained there. A Git push can trigger an existing Render auto-deploy, but live deployment must be verified separately.

For new deployments, use these repository-root settings:

- Root directory: empty (repository root), replacing `deploy/mikteeng-chatbot`.
- Build command: `pip install .`
- Start command: `python -m mikteeng_rspm serve --host 0.0.0.0 --checkpoint models/synthetic_vector_demo.json`
- Health path: `/health`
- Python: 3.12
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`

The server uses Render's `PORT`. It serves a vector prediction console, not the retired text/audio chatbot. Replace the checkpoint path with your own trained RSPM checkpoint to deploy your trained model. A successful `/health` response identifies `Mikteeng RSPM`, version `0.2.0`, dimension 32 for the demo, and `training_enabled: false`.

Train offline or in a persistent compute environment, then deploy the resulting checkpoint. Render's ephemeral filesystem is not a durable training store. No API key is required for this inference implementation. Training is available through the CLI only.

Rollback uses the prior Git commit and restores its corresponding Render commands and root directory.
