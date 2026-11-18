"""Mesure de performance (guide L1.9) : volume du guide collecté en entier, sous le budget."""

from __future__ import annotations

from packages.collect import bench


def test_guide_volume_is_fully_collected_within_budget() -> None:
    report = bench.measure(runs=2)
    assert report["items"] == bench.VOLUME  # 500 mails, 20 événements, 30 tâches, 100 fichiers
    assert report["p95"] < bench.BUDGET_S


def test_latency_is_simulated_per_request() -> None:
    world = bench.build_world()
    send = bench.FakeGraph(world, latency=0.001)
    send("https://graph.microsoft.com/v1.0/me", {})
    assert send.calls == 1
    assert send("https://graph.microsoft.com/v1.0/inconnu", {})[0] == 404


def test_command_reports_p50_and_p95(capsys) -> None:
    assert bench.main(["--runs", "1"]) == 0
    assert "p95" in capsys.readouterr().out
