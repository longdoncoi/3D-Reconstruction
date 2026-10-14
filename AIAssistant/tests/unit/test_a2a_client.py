"""Client-side coverage for the trusted A2A transport.

``TrustedA2AClient`` only talks to deployment-configured endpoints: anything
else is a policy rejection, and network failures degrade to empty results
rather than raising. (Moved verbatim from ``test_coverage_boost.py`` during
the coverage-file split; submit paths from ``test_coverage_boost2.py``.)
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.a2a_client import RemoteAgentCard, TrustedA2AClient
from ai_assistant.domain.security import DataClassification


class TrustedA2AClientTests(unittest.TestCase):
    """Test TrustedA2AClient coverage."""

    def test_client_rejects_untrusted_endpoint(self) -> None:
        """Client rejects endpoints not in allowlist."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )
        result = client.submit("https://untrusted.example.com", "supervisor", "test")
        self.assertEqual(result["routing"], "rejected_policy")

    def test_client_accepts_trusted_endpoint(self) -> None:
        """Client accepts endpoints in allowlist."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )
        self.assertIsNotNone(client)

    def test_client_discover_returns_empty_on_network_failure(self) -> None:
        """discover returns empty dict when network fails."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://unreachable.example.com"}),
            timeout_seconds=1,
        )
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("Network unreachable")
            cards = client.discover()
        self.assertEqual(cards, {})



# ── Submit and discovery paths (moved from test_coverage_boost2) ──

class TrustedA2AClientSubmitTests(unittest.TestCase):
    """Coverage for TrustedA2AClient submit flow."""

    def _client(self) -> TrustedA2AClient:
        return TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )

    def test_submit_happy_path(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"supervisor"}), streaming=False,
        )
        response_body = b'{"task": {"id": "t-1", "status": {"state": "completed"}}}'
        with patch.object(client, "_read_card", return_value=card), patch(
            "ai_assistant.adapters.a2a_client.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = response_body
            mock_urlopen.return_value = mock_resp

            result = client.submit("https://trusted.example.com", "supervisor", "hello")

        self.assertEqual(result["routing"], "remote_selected")
        self.assertEqual(result["task"]["id"], "t-1")

    def test_submit_rejects_restricted_classification(self) -> None:
        client = self._client()
        result = client.submit(
            "https://trusted.example.com", "supervisor", "secret",
            classification=DataClassification.RESTRICTED,
        )
        self.assertEqual(result["routing"], "rejected_policy")

    def test_submit_rejects_missing_capability(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"research"}), streaming=False,
        )
        with patch.object(client, "_read_card", return_value=card):
            result = client.submit("https://trusted.example.com", "supervisor", "hello")
        self.assertEqual(result["routing"], "rejected_no_eligible_agent")

    def test_submit_transport_failure_degraded(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"supervisor"}), streaming=False,
        )
        with patch.object(client, "_read_card", return_value=card), patch(
            "ai_assistant.adapters.a2a_client.urlopen",
            side_effect=OSError("network down"),
        ):
            result = client.submit("https://trusted.example.com", "supervisor", "hello")
        self.assertEqual(result["routing"], "degraded_dependency")

    def test_discover_skips_failed_endpoints(self) -> None:
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://a.example.com", "https://b.example.com"}),
            timeout_seconds=5,
        )
        good = RemoteAgentCard(
            endpoint="https://a.example.com", name="A", version="1", skills=frozenset(), streaming=True,
        )

        def _read_card(endpoint: str) -> RemoteAgentCard:
            if endpoint.endswith("b.example.com"):
                msg = "bad card"
                raise ValueError(msg)
            return good

        with patch.object(client, "_read_card", side_effect=_read_card):
            cards = client.discover()
        self.assertEqual(set(cards), {"https://a.example.com"})

    def test_read_card_invalid_payload(self) -> None:
        client = self._client()
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = b'{"name": 123}'
            mock_urlopen.return_value = mock_resp
            with self.assertRaises(ValueError):
                client._read_card("https://trusted.example.com")

    def test_read_card_no_capabilities(self) -> None:
        client = self._client()
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = (
                b'{"name": "A", "skills": [{"id": 123}]}'
            )
            mock_urlopen.return_value = mock_resp
            with self.assertRaises(ValueError):
                client._read_card("https://trusted.example.com")


class TraceContextPropagationTests(unittest.TestCase):
    """Outbound A2A calls carry the active trace via W3C ``traceparent``.

    This is the outbound half of propagation: when a span is active the header
    is present and well-formed; when there is nothing valid to propagate the
    header is omitted rather than faked. (Inbound extraction on our own A2A
    routes is not wired yet.)
    """

    _EXPECTED = "00-" + "ab" * 16 + "-" + "cd" * 8 + "-01"

    def _capture_headers(self) -> dict[str, str]:
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}), timeout_seconds=5,
        )
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"supervisor"}), streaming=False,
        )
        with patch.object(client, "_read_card", return_value=card), patch(
            "ai_assistant.adapters.a2a_client.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = b'{"task": {"id": "t-1"}}'
            mock_urlopen.return_value = mock_resp
            client.submit("https://trusted.example.com", "supervisor", "hello")
        request = mock_urlopen.call_args[0][0]
        return {key.lower(): value for key, value in request.header_items()}

    def test_an_active_span_adds_traceparent(self) -> None:
        with patch(
            "ai_assistant.adapters.a2a_client.current_traceparent", return_value=self._EXPECTED
        ):
            headers = self._capture_headers()
        self.assertEqual(headers["traceparent"], self._EXPECTED)

    def test_nothing_to_propagate_omits_the_header(self) -> None:
        with patch(
            "ai_assistant.adapters.a2a_client.current_traceparent", return_value=None
        ):
            headers = self._capture_headers()
        self.assertNotIn("traceparent", headers)


if __name__ == "__main__":
    unittest.main()
