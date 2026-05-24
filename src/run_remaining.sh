#!/bin/bash
# Run register and typo probes in parallel.
set -e
cd /workspaces/llm-reading-modes-c01e-claude
PY=./.venv/bin/python

$PY src/probe.py --modes register --layer_stride 2 --output_suffix _register > logs/probe_register.log 2>&1 &
PID1=$!

$PY src/probe.py --modes typo --layer_stride 2 --output_suffix _typo > logs/probe_typo.log 2>&1 &
PID2=$!

echo "Started: register=$PID1 typo=$PID2"
wait $PID1 $PID2
echo "Done."
