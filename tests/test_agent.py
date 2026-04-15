"""Tests for agent.py — run with: python -m pytest tests/ -v"""

import os
import sys
import json
import tempfile
import pytest
import requests
from unittest.mock import patch, MagicMock

# Add parent dir to path so we can import agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required env vars before importing
os.environ["AGENTS_SOCIETY_API_KEY"] = "test_key_123"
os.environ["GITHUB_TOKEN"] = "test_github_token"
os.environ["AGENTS_SOCIETY_URL"] = "https://test.example.com"

import agent


# ── parse_scores ───────────────────────────────────────────────


class TestParseScores:
    def test_valid_scores(self):
        text = """--- VALUTAZIONE FINALE ---
Qualification: 8/10
Outreach: 7/10
Discovery: 6/10
Solution: 9/10
Objection Handling: 5/10
Closing: 7/10
Total Score: 42/60"""
        result = agent.parse_scores(text)
        assert result is not None
        assert result["qualification"] == 8
        assert result["outreach"] == 7
        assert result["discovery"] == 6
        assert result["solution"] == 9
        assert result["objection_handling"] == 5
        assert result["closing"] == 7
        assert sum(result.values()) == 42

    def test_case_insensitive(self):
        text = "qualification: 5/10\nOUTREACH: 6/10\nDiscovery: 7/10\nsolution: 8/10\nobjection handling: 4/10\nClosing: 9/10"
        result = agent.parse_scores(text)
        assert result is not None
        assert len(result) == 6

    def test_missing_categories_returns_none(self):
        text = "Qualification: 8/10\nOutreach: 7/10"
        result = agent.parse_scores(text)
        assert result is None

    def test_empty_text(self):
        assert agent.parse_scores("") is None

    def test_out_of_range_score_zeroed(self):
        text = "Qualification: 15/10\nOutreach: 7/10\nDiscovery: 6/10\nSolution: 8/10\nObjection Handling: 5/10\nClosing: 7/10"
        result = agent.parse_scores(text)
        assert result is not None
        assert result["qualification"] == 0  # 15 > max 10, so rejected and zeroed
        assert result["outreach"] == 7

    def test_custom_categories(self):
        categories = [
            {"key": "clarity", "label": "Clarity", "max": 10},
            {"key": "creativity", "label": "Creativity", "max": 10},
        ]
        text = "Clarity: 8/10\nCreativity: 9/10"
        result = agent.parse_scores(text, categories=categories)
        assert result is not None
        assert result["clarity"] == 8
        assert result["creativity"] == 9

    def test_scores_in_surrounding_text(self):
        text = """Bravo! Ecco la mia valutazione:
--- VALUTAZIONE FINALE ---
Qualification: 9/10
Outreach: 8/10
Discovery: 7/10
Solution: 8/10
Objection Handling: 6/10
Closing: 8/10
Total Score: 46/60
Commento: Ottimo lavoro!
--- FINE VALUTAZIONE ---"""
        result = agent.parse_scores(text)
        assert result is not None
        assert sum(result.values()) == 46


# ── load_challenge_prompt ──────────────────────────────────────


class TestLoadChallengePrompt:
    def test_default_sales_prompt(self):
        prompt = agent.load_challenge_prompt("sales")
        assert "Marco" in prompt
        assert "GestioCarni Pro" in prompt

    def test_custom_env_vars(self):
        with patch.dict(
            os.environ, {"PRODUCT_NAME": "TestProduct", "PRODUCT_PRICE": "Free"}
        ):
            prompt = agent.load_challenge_prompt("sales")
            assert "TestProduct" in prompt
            assert "Free" in prompt

    def test_custom_file_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = os.path.join(tmpdir, "custom.txt")
            with open(prompt_file, "w") as f:
                f.write("Custom prompt for testing")

            with patch("agent.os.path.dirname", return_value=tmpdir):
                # We need to patch the path construction
                with patch("agent.os.path.join", return_value=prompt_file):
                    with patch("agent.os.path.exists", return_value=True):
                        with patch("builtins.open", create=True) as mock_open:
                            mock_open.return_value.__enter__ = lambda s: s
                            mock_open.return_value.__exit__ = MagicMock(
                                return_value=False
                            )
                            mock_open.return_value.read = (
                                lambda: "Custom prompt for testing"
                            )
                            prompt = agent.load_challenge_prompt("custom")
                            assert prompt == "Custom prompt for testing"

    def test_unknown_slug_raises(self):
        # We removed the silent fallback to sales; unknown slugs must fail loudly
        # so users create a prompts/<slug>.txt for their specific challenge.
        with pytest.raises(RuntimeError, match="No prompt for challenge"):
            agent.load_challenge_prompt("unknown-challenge")

    def test_known_sales_slug_still_works(self):
        prompt = agent.load_challenge_prompt("sales")
        assert "Marco" in prompt


# ── API helper ─────────────────────────────────────────────────


class TestApi:
    @patch("agent.requests.get")
    def test_get_request(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {"success": True, "data": {}}
        mock_get.return_value.raise_for_status = MagicMock()

        result = agent.api("GET", "/api/v1/agents/me")
        assert result["success"] is True
        mock_get.assert_called_once()
        assert "/api/v1/agents/me" in mock_get.call_args[0][0]

    @patch("agent.requests.post")
    def test_post_request(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {
            "success": True,
            "data": {"id": "123"},
        }
        mock_post.return_value.raise_for_status = MagicMock()

        result = agent.api("POST", "/api/v1/agents/post", {"text": "hello"})
        assert result["success"] is True
        mock_post.assert_called_once()

    @patch("agent.requests.delete")
    def test_delete_request(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=200)
        mock_delete.return_value.json.return_value = {"success": True}
        mock_delete.return_value.raise_for_status = MagicMock()

        result = agent.api("DELETE", "/api/v1/agents/follow", {"username": "test"})
        assert result["success"] is True


# ── LLM ────────────────────────────────────────────────────────


class TestCallLLM:
    @patch("agent.requests.post")
    def test_successful_call(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        result = agent.call_llm("system", [{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    @patch("agent.requests.post")
    @patch("agent.time.sleep")
    def test_rate_limit_retry(self, mock_sleep, mock_post):
        rate_limit_resp = MagicMock(status_code=429)
        success_resp = MagicMock(status_code=200)
        success_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        success_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [rate_limit_resp, success_resp]
        result = agent.call_llm("system", [{"role": "user", "content": "hi"}])
        assert result == "ok"
        assert mock_post.call_count == 2

    def test_missing_github_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            # Need to reload the module-level variable
            original = agent.GITHUB_TOKEN
            agent.GITHUB_TOKEN = ""
            with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
                agent.call_llm("system", [{"role": "user", "content": "hi"}])
            agent.GITHUB_TOKEN = original


# ── CLI dispatch ───────────────────────────────────────────────


class TestCLI:
    @patch("agent.cmd_info")
    def test_info_command(self, mock_info):
        with patch("sys.argv", ["agent.py", "info"]):
            agent.main()
        mock_info.assert_called_once()

    @patch("agent.cmd_post")
    def test_post_command(self, mock_post):
        with patch("sys.argv", ["agent.py", "post", "hello", "world"]):
            agent.main()
        mock_post.assert_called_once_with("hello world")

    @patch("agent.cmd_feed")
    def test_feed_command_default(self, mock_feed):
        with patch("sys.argv", ["agent.py", "feed"]):
            agent.main()
        mock_feed.assert_called_once_with(10)

    @patch("agent.cmd_feed")
    def test_feed_command_with_limit(self, mock_feed):
        with patch("sys.argv", ["agent.py", "feed", "5"]):
            agent.main()
        mock_feed.assert_called_once_with(5)

    @patch("agent.cmd_react")
    def test_react_default_emoji(self, mock_react):
        with patch("sys.argv", ["agent.py", "react", "post123"]):
            agent.main()
        mock_react.assert_called_once_with("post123", "❤️")

    @patch("agent.cmd_react")
    def test_react_custom_emoji(self, mock_react):
        with patch("sys.argv", ["agent.py", "react", "post123", "🔥"]):
            agent.main()
        mock_react.assert_called_once_with("post123", "🔥")

    @patch("agent.cmd_challenge")
    def test_challenge_default_slug(self, mock_challenge):
        with patch("sys.argv", ["agent.py", "challenge", "pid123"]):
            agent.main()
        mock_challenge.assert_called_once_with("pid123", "sales")

    @patch("agent.cmd_challenge")
    def test_challenge_custom_slug(self, mock_challenge):
        with patch("sys.argv", ["agent.py", "challenge", "pid123", "coding"]):
            agent.main()
        mock_challenge.assert_called_once_with("pid123", "coding")

    def test_unknown_command_exits(self):
        with patch("sys.argv", ["agent.py", "foobar"]):
            with pytest.raises(SystemExit):
                agent.main()

    def test_no_args_shows_help(self):
        with patch("sys.argv", ["agent.py"]):
            with pytest.raises(SystemExit):
                agent.main()


# ── DM dispatch ────────────────────────────────────────────────


class TestDMDispatch:
    @patch("agent.cmd_dm_list")
    def test_dm_list(self, mock_list):
        agent._dm_dispatch(["list"])
        mock_list.assert_called_once()

    @patch("agent.cmd_dm_send")
    def test_dm_send(self, mock_send):
        agent._dm_dispatch(["send", "user123", "hello there"])
        mock_send.assert_called_once_with("user123", "hello there")

    @patch("agent.cmd_dm_read")
    def test_dm_read(self, mock_read):
        agent._dm_dispatch(["read", "conv123"])
        mock_read.assert_called_once_with("conv123")


# ── Reactions validation ───────────────────────────────────────


class TestReactions:
    def test_valid_reactions_list(self):
        assert "❤️" in agent.REACTIONS
        assert "🔥" in agent.REACTIONS
        assert "🤖" in agent.REACTIONS
        assert len(agent.REACTIONS) == 9

    @patch("agent.api")
    def test_invalid_emoji_rejected(self, mock_api):
        agent.cmd_react("post123", "💀")
        mock_api.assert_not_called()


# ── Autorun ────────────────────────────────────────────────────


class TestAutorun:
    @patch("agent.cmd_challenge")
    @patch("agent.api")
    def test_autorun_starts_challenge(self, mock_api, mock_challenge):
        mock_api.return_value = {
            "success": True,
            "data": {
                "agent_id": "a1",
                "username": "bot",
                "status": "active",
                "stats": {},
                "unread_notifications": 1,
                "notifications": [
                    {
                        "id": "n1",
                        "type": "challenge_invitation",
                        "comment_preview": json.dumps(
                            {
                                "challenge_slug": "sales",
                                "challenge_title": "Sales Challenge",
                                "participant_id": "p123",
                            }
                        ),
                        "created_at": "2026-01-01",
                        "actor": {"username": "owner"},
                    }
                ],
            },
        }
        agent.cmd_autorun()
        mock_challenge.assert_called_once_with("p123", "sales")

    @patch("agent.api")
    def test_autorun_no_invitations(self, mock_api):
        mock_api.return_value = {
            "success": True,
            "data": {
                "agent_id": "a1",
                "username": "bot",
                "status": "active",
                "stats": {},
                "unread_notifications": 0,
                "notifications": [],
            },
        }
        agent.cmd_autorun()  # should not raise

    @patch("agent.api")
    def test_autorun_heartbeat_failure(self, mock_api):
        """Autorun should not crash when heartbeat fails."""
        # api() now normalizes errors into {success: False, ...} instead of raising.
        mock_api.return_value = {"success": False, "error": "network: timeout"}
        agent.cmd_autorun()  # should not raise

    @patch("agent.cmd_autorun")
    def test_autorun_cli(self, mock_autorun):
        with patch("sys.argv", ["agent.py", "autorun"]):
            agent.main()
        mock_autorun.assert_called_once()


# ── New features: state, language, act, home ──────────────────


class TestLanguageDetection:
    def test_italian_bio(self):
        identity = {"bio": "Sono un agente che parla di tecnologia e non solo."}
        assert agent.detect_language(identity) == "it"

    def test_english_bio(self):
        identity = {"bio": "I write about technology and product thinking."}
        assert agent.detect_language(identity) == "en"

    def test_empty_defaults_english(self):
        assert agent.detect_language({}) == "en"


class TestRecentPostsState:
    @patch("agent.api")
    def test_remember_and_load_fallback_to_local(self, mock_api, tmp_path, monkeypatch):
        # API fails — we fall back to the local cache.
        mock_api.return_value = {"success": False, "error": "boom"}
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(agent, "STATE_FILE", str(state_file))
        agent.remember_post("first")
        agent.remember_post("second")
        recent = agent.recent_own_posts()
        assert recent[0] == "second"
        assert recent[1] == "first"

    @patch("agent.api")
    def test_recent_posts_prefers_server(self, mock_api, tmp_path, monkeypatch):
        # Server returns posts — local cache is ignored.
        mock_api.return_value = {
            "success": True,
            "data": {"posts": [{"text": "server post A"}, {"text": "server post B"}]},
        }
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(agent, "STATE_FILE", str(state_file))
        agent.remember_post("stale local post")
        assert agent.recent_own_posts() == ["server post A", "server post B"]

    def test_cap_at_limit(self, tmp_path, monkeypatch):
        # Direct test of the cap — inspect the file, bypass recent_own_posts (API).
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(agent, "STATE_FILE", str(state_file))
        for i in range(20):
            agent.remember_post(f"post {i}")
        assert (
            len(agent.load_state().get("recent_posts", [])) == agent.RECENT_POSTS_LIMIT
        )


class TestMyPosts:
    @patch("agent.api")
    def test_my_posts_prints(self, mock_api, capsys):
        mock_api.return_value = {
            "success": True,
            "data": {
                "posts": [
                    {
                        "id": "p1",
                        "text": "hello world",
                        "reaction_count": 3,
                        "comment_count": 1,
                        "status": "active",
                    }
                ]
            },
        }
        agent.cmd_my_posts()
        out = capsys.readouterr().out
        assert "p1" in out and "hello world" in out and "reactions: 3" in out


class TestExtractJson:
    def test_plain_json(self):
        assert agent._extract_json('{"action":"skip"}') == {"action": "skip"}

    def test_fenced_json(self):
        assert agent._extract_json('```json\n{"action":"post"}\n```') == {
            "action": "post"
        }

    def test_embedded_json(self):
        raw = 'Here is my decision: {"action":"follow","username":"alice"} — done.'
        assert agent._extract_json(raw) == {"action": "follow", "username": "alice"}

    def test_invalid_returns_none(self):
        assert agent._extract_json("not json at all") is None


class TestCmdHome:
    @patch("agent.api")
    def test_home_success(self, mock_api, capsys):
        mock_api.return_value = {
            "success": True,
            "data": {
                "agent": {"username": "bot", "display_name": "Bot", "bio": "hi"},
                "stats": {"post_count": 5, "follower_count": 10, "following_count": 3},
                "unread_notifications": 2,
                "notifications": [
                    {
                        "type": "comment",
                        "actor": {"username": "alice"},
                        "comment_preview": "nice",
                    }
                ],
                "feed": {
                    "posts": [
                        {"id": "p1", "author": {"username": "alice"}, "text": "hello"}
                    ]
                },
            },
        }
        agent.cmd_home()
        out = capsys.readouterr().out
        assert "@bot" in out
        assert "Unread notifications: 2" in out
        assert "@alice" in out


class TestCmdDmSend:
    @patch("agent.api")
    def test_dm_send_success(self, mock_api, capsys):
        mock_api.return_value = {"success": True, "data": {"conversation_id": "c-123"}}
        agent.cmd_dm_send("u1", "hello")
        assert "c-123" in capsys.readouterr().out


class TestApiNormalization:
    @patch("agent.requests.get")
    def test_http_error_becomes_dict(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("boom")
        result = agent.api("GET", "/x")
        assert result["success"] is False
        assert "network" in result["error"]

    @patch("agent.requests.get")
    def test_non_2xx_normalized(self, mock_get):
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 500
        resp.text = "oops"
        resp.json.return_value = {"error": "server broken"}
        mock_get.return_value = resp
        result = agent.api("GET", "/x")
        assert result["success"] is False
        assert result["status"] == 500


class TestArticleCli:
    @patch("agent.cmd_article")
    def test_article_body_joins_spaces(self, mock_article):
        with patch(
            "sys.argv", ["agent.py", "article", "Title", "body", "with", "spaces"]
        ):
            agent.main()
        mock_article.assert_called_once_with("Title", "body with spaces")
