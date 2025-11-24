#!/usr/bin/env bash

uv run coverage run --source=atria_logger -m pytest $@ 
uv run coverage report --show-missing
