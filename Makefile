.PHONY: install lint test train run-api run-ui load-snowflake docker-up docker-down

install:
	pip install -r requirements/dev.txt

lint:
	ruff check .

test:
	pytest -v --cov=src --cov=api --cov-report=term-missing

train:
	python -m src.models.train

load-snowflake:
	python -m src.data.load_to_snowflake

run-api:
	uvicorn api.main:app --reload --port 8000

run-ui:
	streamlit run streamlit_app/Home_Page.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
