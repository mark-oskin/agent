"""Lane label matching (fork dedupe uses the same rules as agent_send routing)."""


def lanes_matching_name(labels: list[str], name: str) -> list[int]:
    key = name.casefold().strip()
    return [i for i, lab in enumerate(labels) if lab.casefold().strip() == key]


def test_duplicate_worker_label_matches_one_lane():
    labels = ["Agent 1", "worker"]
    assert lanes_matching_name(labels, "worker") == [1]
    assert lanes_matching_name(labels, "Worker") == [1]


def test_duplicate_labels_in_list():
    labels = ["Agent 1", "worker", "worker"]
    assert lanes_matching_name(labels, "worker") == [1, 2]
