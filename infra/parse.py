"""Parse `vastai --raw` JSON on stdin. One tiny verb per line of vast.sh.

Kept separate from vast.sh so the JSON handling is testable and so we are not
depending on jq being installed.
"""

from __future__ import annotations

import json
import sys


def load():
    text = sys.stdin.read().strip()
    if not text:
        return None
    return json.loads(text)


def as_instance(data):
    """`show instance` returns a dict; some versions wrap it in a list."""
    if isinstance(data, list):
        if not data:
            sys.exit("no instance in response")
        return data[0]
    if isinstance(data, dict) and "instances" in data:
        return data["instances"][0]
    return data


def offers(data) -> None:
    """Human-readable table of candidate offers."""
    rows = data or []
    print(f"{'ID':>10}  {'$/hr':>6}  {'GPU':<16} {'RAM':>5} {'disk':>6} {'net↓':>6}  loc")
    for offer in rows[:15]:
        print(
            f"{offer.get('id', '?'):>10}  "
            f"{offer.get('dph_total', 0):>6.3f}  "
            f"{offer.get('gpu_name', '?')!s:<16} "
            f"{offer.get('cpu_ram', 0) / 1024:>4.0f}G "
            f"{offer.get('disk_space', 0):>5.0f}G "
            f"{offer.get('inet_down', 0):>5.0f}M  "
            f"{offer.get('geolocation', '?')}"
        )


def cheapest(data, field: str) -> None:
    rows = data or []
    if not rows:
        return  # empty output; vast.sh treats that as "no offer"
    best = min(rows, key=lambda o: o.get("dph_total", 9e9))
    print(best["id"] if field == "id" else f"{best.get('dph_total', 0):.4f}")


def new_id(data) -> None:
    """`create instance --raw` returns {"success": true, "new_contract": N}."""
    if not data or not data.get("success"):
        sys.exit(f"create failed: {data}")
    print(data["new_contract"])


def status(data) -> None:
    print(as_instance(data).get("actual_status") or "pending")


def hostport(data) -> None:
    """Prefer the direct-SSH ip/port; fall back to vast's ssh proxy."""
    inst = as_instance(data)
    ip = inst.get("public_ipaddr")
    ports = inst.get("ports") or {}
    mapped = ports.get("22/tcp") or []
    if ip and mapped:
        print(f"{str(ip).strip()} {mapped[0]['HostPort']}")
        return
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    if not (host and port):
        sys.exit("could not determine ssh host/port from instance json")
    print(f"{host} {port}")


def dph(data):
    """The instance's real hourly rate, which differs from the offer's quote."""
    inst = data[0] if isinstance(data, list) and data else data
    v = inst.get("dph_total") if isinstance(inst, dict) else None
    if v is not None:
        print(v)


def has_instance(data, wanted):
    """Exit 0 if `wanted` is still on the account, 1 otherwise.

    Used by `down` to confirm a destroy actually happened -- `vastai destroy`
    aborts its own [y/N] prompt without a failing exit code, so its return
    value proves nothing.
    """
    items = data if isinstance(data, list) else [data]
    ids = {str(i.get("id")) for i in items if isinstance(i, dict)}
    sys.exit(0 if str(wanted) in ids else 1)


VERBS = {
    "offers": offers,
    "cheapest-id": lambda d: cheapest(d, "id"),
    "cheapest-price": lambda d: cheapest(d, "price"),
    "new-id": new_id,
    "status": status,
    "hostport": hostport,
    "has-instance": has_instance,
    "dph": dph,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in VERBS:
        sys.exit(f"usage: parse.py {{{'|'.join(VERBS)}}} [arg]")
    VERBS[sys.argv[1]](load(), *sys.argv[2:])
