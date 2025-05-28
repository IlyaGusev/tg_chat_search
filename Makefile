.PHONY: black style validate test install serve

install:
	uv pip install -e .

black:
	uv run black scripts --line-length 100
	uv run black chat_search --line-length 100

validate:
	uv run black scripts --line-length 100
	uv run black chat_search --line-length 100
	uv run flake8 scripts
	uv run flake8 chat_search
	uv run mypy scripts --strict --explicit-package-bases
	uv run mypy chat_search --strict --explicit-package-bases

test:
	uv run pytest -s

serve:
	uv run python -m chat_search.main
