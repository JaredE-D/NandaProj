"""Tests for infra/parse.py -- the vastai JSON shapes we depend on.

These are the fields that break silently when vastai changes its output, so
they are worth pinning.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PARSE = Path(__file__).resolve().parents[1] / "infra" / "parse.py"


def run(verb: str, payload) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PARSE), verb],
        input=json.dumps(payload),
        check=False,
        capture_output=True,
        text=True,
    )


OFFER = {
    "id": 123,
    "dph_total": 0.29,
    "gpu_name": "RTX 4090",
    "cpu_ram": 32000,
    "disk_space": 80,
    "inet_down": 900,
    "geolocation": "PL",
}


def test_cheapest_picks_lowest_price():
    cheap = dict(OFFER, id=999, dph_total=0.19)
    out = run("cheapest-id", [OFFER, cheap])
    assert out.stdout.strip() == "999"


def test_cheapest_price_is_formatted():
    assert run("cheapest-price", [OFFER]).stdout.strip() == "0.2900"


def test_cheapest_on_empty_list_prints_nothing():
    # vast.sh relies on empty output meaning "no matching offer".
    out = run("cheapest-id", [])
    assert out.stdout.strip() == ""
    assert out.returncode == 0


def test_new_id_extracts_contract():
    assert run("new-id", {"success": True, "new_contract": 55}).stdout.strip() == "55"


def test_new_id_fails_loudly_on_failure():
    out = run("new-id", {"success": False, "error": "no_such_offer"})
    assert out.returncode != 0
    assert "create failed" in out.stderr


def test_status_defaults_to_pending():
    assert run("status", {"actual_status": None}).stdout.strip() == "pending"
    assert run("status", {"actual_status": "running"}).stdout.strip() == "running"


def test_hostport_prefers_direct_ip():
    inst = {
        "public_ipaddr": "1.2.3.4 ",
        "ports": {"22/tcp": [{"HostPort": "41234"}]},
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 12345,
    }
    assert run("hostport", inst).stdout.strip() == "1.2.3.4 41234"


def test_hostport_falls_back_to_ssh_proxy():
    inst = {"public_ipaddr": None, "ports": {}, "ssh_host": "ssh5.vast.ai", "ssh_port": 12345}
    assert run("hostport", inst).stdout.strip() == "ssh5.vast.ai 12345"


def test_hostport_fails_when_unknown():
    out = run("hostport", {"public_ipaddr": None, "ports": {}})
    assert out.returncode != 0


def test_instance_unwrapping_handles_list_form():
    assert run("status", [{"actual_status": "running"}]).stdout.strip() == "running"


@pytest.mark.parametrize("verb", ["offers", "cheapest-id", "status", "hostport", "new-id"])
def test_verbs_exist(verb):
    # A typo in a verb name would only surface at rental time otherwise.
    out = subprocess.run(
        [sys.executable, str(PARSE), verb],
        input="null",
        capture_output=True,
        text=True,
        check=False,
    )
    assert "usage: parse.py" not in out.stderr


class TestHasInstance:
    """`down` trusts this to decide whether a destroy actually happened.

    A false 'gone' means we delete the local instance id while the box keeps
    billing, and `just status` then reports nothing to destroy -- the exact
    failure this verb exists to prevent.
    """

    def _run(self, payload, wanted):
        import json
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [sys.executable, str(root / "infra" / "parse.py"), "has-instance", str(wanted)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    def test_present_in_list_exits_zero(self):
        assert self._run([{"id": 42}, {"id": 7}], 42) == 0

    def test_absent_exits_nonzero(self):
        assert self._run([{"id": 7}], 42) == 1

    def test_empty_account_exits_nonzero(self):
        assert self._run([], 42) == 1

    def test_bare_object_is_accepted(self):
        assert self._run({"id": 42}, 42) == 0

    def test_id_compared_as_string_not_identity(self):
        # vast.ai returns ints; we pass the id in from a file as text.
        assert self._run([{"id": 42}], "42") == 0
