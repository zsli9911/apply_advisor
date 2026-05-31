.PHONY: install db-up db-down ingest chat web test eval

install:
	pip install -r requirements.txt

db-up:
	docker compose up -d

db-down:
	docker compose down

ingest:
	python -m apply_advisor.cli --ingest

chat:
	python -m apply_advisor.cli --verbose

web:
	python -m apply_advisor.webapp

test:
	MOCK_LLM=1 python tests/test_offline.py

eval:
	MOCK_LLM=1 NO_DB=1 python tests/eval_cases.py
