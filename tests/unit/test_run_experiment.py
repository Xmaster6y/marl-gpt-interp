from omegaconf import OmegaConf

from scripts import run_experiment


def test_dispatch_registry_contains_grf_stats_and_not_demo():
    assert run_experiment.SCRIPTS == {
        "grf_rollout_stats": "scripts.grf_rollout_stats:main",
        "normalize_soccer_data": "scripts.normalize_soccer_data:main",
        "compare_soccer_stats": "scripts.compare_soccer_stats:main",
    }


def test_dispatch_runs_exactly_one_selected_script(monkeypatch):
    calls = []

    def fake_load_script(target):
        calls.append(("load", target))

        def fake_main(cfg):
            calls.append(("run", cfg.grf_rollout_stats.name))

        return fake_main

    monkeypatch.setattr(run_experiment, "_load_script", fake_load_script)

    cfg = OmegaConf.create({"grf_rollout_stats": {"name": "smoke"}})
    run_experiment.main.__wrapped__(cfg)

    assert calls == [
        ("load", "scripts.grf_rollout_stats:main"),
        ("run", "smoke"),
    ]
