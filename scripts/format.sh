#!/usr/bin/env bash

ruff check src --fix      # linter
ruff format src --check # formatter
