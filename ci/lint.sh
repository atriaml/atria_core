#!/usr/bin/env bash

mypy src --follow-imports=skip     # type check
ruff check src        # linter
ruff format src --check # formatter
