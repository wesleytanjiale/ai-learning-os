.PHONY: run test eval eval-tune notebook enrich-watchlater import-watchlater

run:
	uv run streamlit run app.py

test:
	uv run pytest tests/ -v -s

eval:
	uv run python evals/run_eval.py

eval-tune:
	uv run python evals/run_eval.py --tune

notebook:
	uv run jupyter notebook

enrich-watchlater:
	uv run python scripts/enrich_watchlater.py --resume

import-watchlater:
	uv run python scripts/import_watchlater.py
