# Release validation

mikteeng AI 0.1.0 was validated against the installed dependency versions in requirements-tested.txt.

- Built a wheel using pip with local setuptools and wheel.
- Installed the wheel into a separate target directory with no dependency downloads.
- Ran from /tmp with only the installed target on PYTHONPATH; the source tree and old experiment module directories were not on the import path.
- All five API tests passed: preserved generation and prefix revision; old sentence/passage heads and character streaming; train/save/load round trip; annotated-role training linked to generation; invalid data/state and unsupported checkpoint-schema rejection.
- The command-line entry point returned the expected dog actor and cat receiver.
- The migrated checkpoint preserves the latest research-model behavior. Its class names are package-qualified and require no legacy import aliases at runtime.

These local checks preceded the GitHub upload. No PyPI publication or external deployment was performed. This is a local installable release, not a claim of production reliability. Runtime peak memory, broad real-world accuracy and cross-version checkpoint compatibility were not benchmarked. Pickle remains trusted-input-only.
