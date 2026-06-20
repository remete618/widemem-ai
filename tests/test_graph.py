"""Unit tests for the optional graph-memory layer (no API/LLM required)."""
from __future__ import annotations

from widemem.graph.extract import parse_triples
from widemem.graph.store import GraphStore


def test_parse_triples_valid_and_normalized():
    out = parse_triples('{"triples": [["Caroline","Moved To","San Francisco"]]}')
    assert out == [("caroline", "moved_to", "san francisco")]


def test_parse_triples_drops_malformed_and_self_loops():
    raw = '{"triples": [["a","r"], ["x","rel","x"], ["good","rel","target"], "junk"]}'
    assert parse_triples(raw) == [("good", "rel", "target")]


def test_parse_triples_bad_json_returns_empty():
    assert parse_triples("not json") == []
    assert parse_triples("") == []


def test_store_add_and_neighbors(tmp_path):
    g = GraphStore(str(tmp_path / "g.db"))
    g.add_triples([("caroline", "moved_from", "sweden")], memory_id="m1", user_id="u")
    assert g.edge_count() == 1
    nb = g.neighbors("caroline", "u")
    assert nb == [("caroline", "moved_from", "sweden", "m1")]
    g.close()


def test_store_filters_noise_anchors(tmp_path):
    g = GraphStore(str(tmp_path / "g.db"))
    n = g.add_triples([("i", "like", "it"), ("caroline", "likes", "pottery")],
                      memory_id="m1", user_id="u")
    assert n == 1  # "i"/"it" edge dropped
    g.close()


def test_expand_surfaces_connected_memories(tmp_path):
    g = GraphStore(str(tmp_path / "g.db"))
    g.add_triples([("caroline", "pursuing", "adoption")], "m1", "u")
    g.add_triples([("adoption", "supports", "lgbtq families")], "m2", "u")
    g.add_triples([("melanie", "has", "dogs")], "m3", "u")
    mem_ids, dist = g.expand(["caroline"], user_id="u", hops=2)
    assert mem_ids == {"m1", "m2"}          # melanie's memory not reached
    assert dist["adoption"] == 1
    assert dist["lgbtq families"] == 2
    g.close()


def test_dates_are_leaves_not_bridges(tmp_path):
    # Two unrelated people both linked to 2019 must NOT connect through it.
    g = GraphStore(str(tmp_path / "g.db"))
    g.add_triples([("caroline", "happened_on", "2019")], "m1", "u")
    g.add_triples([("bob", "happened_on", "2019")], "m2", "u")
    mem_ids, dist = g.expand(["caroline"], user_id="u", hops=3)
    assert "2019" in dist                   # reachable as an endpoint
    assert "bob" not in dist                # but not bridged through the year
    assert mem_ids == {"m1"}
    g.close()


def test_expand_respects_user_scope(tmp_path):
    g = GraphStore(str(tmp_path / "g.db"))
    g.add_triples([("caroline", "likes", "pottery")], "m1", "userA")
    g.add_triples([("caroline", "likes", "running")], "m2", "userB")
    mem_ids, _ = g.expand(["caroline"], user_id="userA", hops=1)
    assert mem_ids == {"m1"}
    g.close()
