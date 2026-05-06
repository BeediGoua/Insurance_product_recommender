import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.agents.tools import run_decisionflow_tool
import json


def test_run_decisionflow_tool_returns_json():
    output = run_decisionflow_tool("C3")
    # Should be JSON serialisable
    result = json.loads(output)
    assert isinstance(result, dict)
    assert "client_profile" in result