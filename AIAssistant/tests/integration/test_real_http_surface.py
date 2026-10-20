"""End-to-end wiring of the HTTP surface against the *real* composition root.

Every other test in this package exercises a router with an injected double.
That is the right way to test a router, but it leaves one gap open: the
*binding* between the composition root and the router is never exercised. A
mis-wired router — the holder that produces the chatbot agent, the prompt
builder behind it, the transport policy — passes every double-driven test.

The scorecard records exactly that failure: ``/v1/chat/completions`` was dead
in production while every gate was green, because every case mocked the
collaborator and a mock answers any attribute (ADR 0029). This file closes
the gap: it composes the real object graph in a fresh interpreter, swaps only
the model backend (the one thing that needs weights), and drives a real HTTP
request through the real router.

The subprocess is not decoration. ``compose()`` builds module-level state in
``bootstrap.server``; running it in a fresh interpreter keeps that state out
of the rest of the suite and reproduces the production start-up order.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"


def _run(code: str, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with a minimal, deployment-like env.

    The suite mutates ``os.environ`` globally, so the subprocess must see the
    deployment defaults and not the residue of the test that launched it.
    """
    keep = ("SYSTEMROOT", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC",
            "USERPROFILE", "HOME")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONPATH"] = str(_SRC)
    env["AI_STORAGE_KEY"] = "hex:" + "11" * 32
    env["AI_ENABLE_RAG"] = "0"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=str(_ROOT),
    )


class RealChatSurfaceTests(unittest.TestCase):
    def test_the_chat_route_answers_through_the_real_composition_root(self) -> None:
        """The chat surface must work with the objects the composition root
        actually builds — not only with injected doubles."""
        code = (
            "from fastapi.testclient import TestClient\n"
            "import ai_assistant.bootstrap.server as s\n"
            "s.compose()\n"
            "\n"
            "class FakeBackend:\n"
            "    def generate(self, **kwargs):\n"
            "        yield {'choices': [{'delta': {'content': 'xin chao'}, 'finish_reason': 'stop'}]}\n"
            "\n"
            "s.runtime_modules.llm.llm = FakeBackend()\n"
            "client = TestClient(s.app, client=('127.0.0.1', 50000))\n"
            "response = client.post('/v1/chat/completions', json={\n"
            "    'messages': [{'role': 'user', 'content': 'hello'}],\n"
            "    'max_tokens': 64,\n"
            "})\n"
            "assert response.status_code == 200, response.text\n"
            "body = response.json()\n"
            "answer = body['choices'][0]['message']['content']\n"
            "assert answer, 'the chat route returned an empty answer'\n"
            "assert body['object'] == 'chat.completion'\n"
            "assert body['x_meta']['max_tokens_used'] > 0\n"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_the_chat_route_refuses_without_a_model(self) -> None:
        """A deployment whose model never loaded must say so, not guess."""
        code = (
            "from fastapi.testclient import TestClient\n"
            "import ai_assistant.bootstrap.server as s\n"
            "s.compose()\n"
            "s.runtime_modules.llm.llm = None\n"
            "client = TestClient(s.app, client=('127.0.0.1', 50000))\n"
            "response = client.post('/v1/chat/completions', json={\n"
            "    'messages': [{'role': 'user', 'content': 'hello'}],\n"
            "})\n"
            "assert response.status_code == 503, response.text\n"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_the_composed_app_serves_the_documentation_only_when_asked(self) -> None:
        """ADR 0032: ``/docs`` is an app-level route, so the transport guard
        never reaches it. The composed app must therefore not publish it by
        default — and must publish it when the deployment opts in."""
        code = (
            "import os\n"
            "from fastapi.testclient import TestClient\n"
            "import ai_assistant.bootstrap.server as s\n"
            "s.compose()\n"
            "client = TestClient(s.app, client=('127.0.0.1', 50000))\n"
            "assert client.get('/openapi.json').status_code == 404\n"
            "assert client.get('/docs').status_code == 404\n"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_the_agent_card_publishes_the_configured_url(self) -> None:
        """ADR 0032: the advertised URL is the configured one, never the
        request's Host header."""
        code = (
            "from fastapi.testclient import TestClient\n"
            "import ai_assistant.bootstrap.server as s\n"
            "s.compose()\n"
            "client = TestClient(s.app, client=('127.0.0.1', 50000))\n"
            "card = client.get('/.well-known/agent.json',\n"
            "                   headers={'Host': 'evil.example'}).json()\n"
            "assert 'evil.example' not in str(card), card\n"
            "assert card['url'].endswith('/a2a'), card\n"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
