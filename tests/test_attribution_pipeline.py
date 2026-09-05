"""The drivers, wired together the way 05 wires them, on a planted ground truth.

`test_attribution.py` covers the arithmetic in isolation. This covers
`patch_sweep` -> `rank` -> `select_set` -> `jaccard` as a pipeline, against a
fake model where exactly two components are known to carry the answer. It is
the test that catches an API mismatch between the pieces -- the failure that
otherwise appears for the first time with a GPU already billing.
"""

from __future__ import annotations

import re

import numpy as np

from nandaproj import attribution as at
from nandaproj import items

N_LAYERS, N_HEADS = 8, 4
PLANTED = {(6, at.HEAD, 2): 0.8, (5, at.MLP, None): 0.5}
NOISE = 0.01


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(text)) % 1000]


class FakePatcher:
    """LD_D = -5, LD_H = +5, and a patch only helps if it came from *this*
    item's honest run -- which is the behaviour the wrong-source null exists
    to detect."""

    n_layers, n_heads = N_LAYERS, N_HEADS

    yes_only = frozenset({at.Component(4, at.HEAD, 1)})

    def __init__(self):
        self.tok = FakeTok()
        self.layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                            for i in range(self.n_layers)]
        self.real = {at.Component(l, k, h, self.layer_types[l]): v
                     for (l, k, h), v in PLANTED.items()}

    def components(self):
        out = []
        for layer in range(self.n_layers):
            out += [at.Component(layer, at.HEAD, h, self.layer_types[layer])
                    for h in range(self.n_heads)]
            out.append(at.Component(layer, at.MLP, None, self.layer_types[layer]))
        return out

    def cache_slot(self, prompt):
        return {c: prompt for c in self.components()}

    @staticmethod
    def _item_of(text):
        m = re.search(r"@(\w+)@", text)
        return m.group(1) if m else None

    def patched_logit_diff(self, prompt, patches, id_h, id_d):
        item = self._item_of(prompt)
        share = 0.0
        for comp, value in patches.items():
            if self._item_of(value) != item:
                continue
            if comp in self.yes_only:
                # A ' Yes'-writer: it helps when ' Yes' is the honest answer and
                # hurts by the same amount when it is the lie. That sign flip is
                # what LD being polarity-signed per item means in practice.
                share += 0.9 if POLARITY_OF.get(item) == at.YES else -0.9
            else:
                share += self.real.get(comp, NOISE)
        return -5.0 + min(max(share, -1.0), 1.0) * 10.0


# Six items, alternating polarity -- three Yes-true and three No-true, the
# balanced shape a whole-pairs gate produces.
POLARITY_OF = {f"I{i}": (at.YES if i % 2 == 0 else at.NO) for i in range(6)}


def fixture(n_items=6):
    patcher = FakePatcher()
    item_list = [items.load([{"id": f"I{i}", "question": f"@I{i}@ q?",
                              "answer": " Yes", "answer_lie": " No"}],
                            kind="records")[f"I{i}"] for i in range(n_items)]
    ld = {it.item_id: {"id_h": 1, "id_d": 2, "H": 5.0, "D": -5.0, "C2": -5.0}
          for it in item_list}
    return patcher, item_list, ld


def swept(condition="D"):
    patcher, item_list, ld = fixture()
    by_id = {it.item_id: it for it in item_list}
    comps = patcher.components()
    pairs = at.derangement([it.item_id for it in item_list], seed=0)
    null = at.patch_sweep(patcher, item_list, ld, condition=condition,
                          components=comps, sources=pairs, by_id=by_id)
    real = at.patch_sweep(patcher, item_list, ld, condition=condition,
                          components=comps, by_id=by_id)
    return patcher, item_list, ld, comps, real, null


def test_the_pipeline_finds_exactly_the_planted_components():
    _, _, _, _, real, null = swept()
    found = {r.component.name for r in at.hits(at.rank(real + null))}
    assert found == {"L6H2", "L5M"}


def test_the_null_sweep_recovers_nothing():
    """If the wrong-source null could recover the answer, it would not be a
    null -- it would be a second measurement of the same thing."""
    _, _, _, _, _, null = swept()
    assert at.recoveries(null).max() == 0.0


def test_a_sweep_writes_after_every_item(tmp_path):
    patcher, item_list, ld = fixture(n_items=3)
    path = tmp_path / "rows.json"
    at.patch_sweep(patcher, item_list, ld, condition="D",
                   components=patcher.components()[:5],
                   by_id={it.item_id: it for it in item_list}, save_to=path)
    assert len(at.load_results(path)) == 3 * 5


def test_a_condition_without_a_baseline_raises_rather_than_using_d():
    """Falling back to D's baseline would make every C2 number wrong and none
    of them obviously so -- and V6 is decided on that comparison."""
    patcher, item_list, ld = fixture(n_items=2)
    for entry in ld.values():
        del entry["C2"]
    try:
        at.patch_sweep(patcher, item_list, ld, condition="C2",
                       components=patcher.components()[:2],
                       by_id={it.item_id: it for it in item_list})
    except KeyError as exc:
        assert "C2" in str(exc)
    else:                                                    # pragma: no cover
        raise AssertionError("silently used the wrong baseline")


def test_select_set_picks_the_strongest_component_first():
    patcher, item_list, ld, _comps, real, null = swept()
    ranked = at.rank(real + null)
    joint = at.joint_recovery_fn(patcher, item_list, ld, condition="D")
    chosen, curve = at.select_set([r.component for r in ranked], joint)
    assert chosen[0].name == "L6H2"
    assert curve[0] == np.float64(0.8) or abs(curve[0] - 0.8) < 1e-9


def test_v6_shape_identical_reads_as_one_and_disjoint_as_zero():
    _patcher, _, _, comps, real, null = swept()
    chosen = [r.component for r in at.hits(at.rank(real + null))]
    disjoint = [at.Component(0, at.HEAD, h, "sliding_attention")
                for h in range(len(chosen))]
    assert at.jaccard(chosen, chosen) == 1.0
    assert at.jaccard(chosen, disjoint) == 0.0
    assert at.overlap_null(comps, len(chosen), n=50, seed=5).mean() < 0.5


# -- polarity arms, end to end ---------------------------------------------

def test_a_planted_yes_writer_is_rejected_by_the_arm_requirement():
    """The pooled median already cancels it on a balanced set; the arms are what
    make that a stated bar rather than an accident of the gate's arithmetic."""
    _, _, _, _, real, null = swept()
    armed_hits = {r.component.name
                  for r in at.hits(at.rank(real + null, polarity_of=POLARITY_OF))}
    assert "L4H1" not in armed_hits
    assert armed_hits == {"L6H2", "L5M"}


def test_the_yes_writer_shows_opposite_signs_in_the_two_arms():
    """Not just 'below the bar' -- the diagnostic is that it recovers in one
    polarity direction and anti-recovers in the other."""
    _, _, _, _, real, null = swept()
    (writer,) = [r for r in at.rank(real + null, polarity_of=POLARITY_OF)
                 if r.component.name == "L4H1"]
    assert writer.median_yes > 0 > writer.median_no
    assert abs(writer.arm_gap) > 1.0


def test_the_planted_belief_components_are_symmetric_across_arms():
    _, _, _, _, real, null = swept()
    for r in at.hits(at.rank(real + null, polarity_of=POLARITY_OF)):
        assert abs(r.arm_gap) < 1e-9


def test_an_unbalanced_set_promotes_the_yes_writer_unless_the_arms_are_required():
    """The failure the arm bar exists for, reproduced.

    On a balanced set the ' Yes'-writer's two arms cancel and the pooled median
    is 0.0 -- 05 is insulated for free. But MIN_LD_GAP drops single twins, and
    one dropped No-true item is enough to tilt the pooled median over the bar.
    Pooled promotes it; the arms still refuse it.
    """
    _, _, _, _, real, null = swept()
    dropped = "I1"                                   # one No-true twin, gate-dropped
    rows = [r for r in real + null if r.item_id != dropped]

    pooled = {r.component.name for r in at.hits(at.rank(rows))}
    armed_hits = {r.component.name
                  for r in at.hits(at.rank(rows, polarity_of=POLARITY_OF))}

    assert "L4H1" in pooled, "the unbalanced pooled median should promote the writer"
    assert "L4H1" not in armed_hits
