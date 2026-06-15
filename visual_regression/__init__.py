"""Visual-regression benchmark harness for OpenRetailScience plots.

This package generates a labelled dataset of deliberately-broken charts (built with the real
``openretailscience.plots`` API and then mutated with matplotlib) and provides a harness for asking a
vision LLM — Claude Haiku via ``claude -p`` or an open model via OpenRouter — to detect the injected
visual regressions. See ``README.md`` for the end-to-end workflow.
"""
