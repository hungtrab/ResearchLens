.PHONY: install run test collect evaluate figures documents verify solr solr-index docker clean
install:
	uv sync --frozen --extra dev
run:
	uv run uvicorn researchlens.api:app --host 127.0.0.1 --port 8765
test:
	uv run pytest -q --cov=researchlens --cov-report=term-missing
collect:
	uv run python scripts/collect.py --per-topic 400
conferences:
	uv run python scripts/crawl_conferences.py --from-year 2020 --to-year 2026
evaluate:
	uv run python scripts/evaluate.py --tasks 120
figures:
	uv run python scripts/figures.py
	cp artifacts/figures/*_table.tex report/generated/
documents: figures
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd report/main.tex
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd slides/main.tex
	mkdir -p deliverables
	cp report/main.pdf deliverables/ResearchLens_Report.pdf
	cp slides/main.pdf deliverables/ResearchLens_Slides.pdf
verify:
	uv run python scripts/verify_artifacts.py
solr:
	docker compose --profile solr up -d solr
solr-index:
	uv run python scripts/solr_index.py
docker:
	docker compose up -d --build app
clean:
	latexmk -c -cd report/main.tex
	latexmk -c -cd slides/main.tex
