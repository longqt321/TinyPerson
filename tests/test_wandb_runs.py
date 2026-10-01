from types import SimpleNamespace

from tinydet.utils.wandb_runs import tag_latest_runs


def test_latest_per_model_preserves_history_and_other_tags():
    changed = []
    def run(identifier, name, tags):
        return SimpleNamespace(id=identifier, config={"name": name}, tags=tags,
                               update=lambda: changed.append(identifier))
    runs = [run("new", "p2", ["experiment"]), run("other", "p3", []),
            run("old", "p2", ["latest", "keep"]), run("unknown", None, [])]
    api = SimpleNamespace(runs=lambda *args, **kwargs: runs)
    result = tag_latest_runs(api, "team/project")
    assert result["latest"] == {"p2": "new", "p3": "other"}
    assert runs[0].tags == ["experiment", "latest"]
    assert runs[2].tags == ["keep", "historical"]
    assert len(changed) == 3
    assert tag_latest_runs(api, "team/project")["updated"] == 0
