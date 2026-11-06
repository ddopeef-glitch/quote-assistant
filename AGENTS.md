# Development constraints

- This is a synthetic portfolio edition. Keep company/customer materials, credentials, generated workbooks and runtime records out of Git and release archives.
- Business invariants and current boundaries are in `docs/BEHAVIOR.md`. Do not infer unknown prices, packaging, grades or product identities.
- `app.py` owns the UI; `src/` owns extraction, validation, pricing and rendering. Reuse the existing confirmation and fingerprint checks.
- Run `python scripts/create_demo_data.py` and `python -m pytest -q` after core changes. Tests must not make network requests or load real OCR models.
- Use `scripts/package_release.py` to verify the reviewed `public-manifest.json`. Review changed files before updating hashes; images need visual inspection. Never import old repository history.
- Do not describe offline replay as model inference, or local tests as production/customer acceptance.
