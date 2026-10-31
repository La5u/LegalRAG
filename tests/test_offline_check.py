import pytest

from scripts.offline_check import evaluate_response


def response_events(answer="하루 1천분의 2.5, 최대 100분의 8입니다."):
    return [
        {"type": "sources", "sources": [{
            "raw_document": "documents/raw/samples/샘플_소프트웨어개발용역계약서.txt",
            "text": "제9조(지체상금) 하루 1천분의 2.5, 최대 100분의 8.",
        }]},
        {"type": "token", "text": answer},
        {"type": "done"},
    ]


@pytest.mark.parametrize("answer", [
    "하루 1천분의 2.5, 최대 100분의 8입니다.", "하루 0.25%, 최대 8%입니다.",
])
def test_sample_response_passes(answer):
    result = evaluate_response(response_events(answer))
    assert all(result["checks"].values())
    assert result["answer"] == answer
    assert "제9조" in result["clauses"][0]


@pytest.mark.parametrize("fault", ["source", "clause", "rate", "cap", "error", "incomplete"])
def test_incorrect_or_incomplete_response_fails(fault):
    events = response_events()
    if fault == "source":
        events[0]["sources"][0]["raw_document"] = "another.txt"
    elif fault == "clause":
        events[0]["sources"][0]["text"] = "제1조 목적"
    elif fault == "rate":
        events[1]["text"] = "최대 8%"
    elif fault == "cap":
        events[1]["text"] = "하루 0.25%"
    elif fault == "error":
        events.append({"type": "error", "message": "generation failed"})
    else:
        events.pop()
    assert not all(evaluate_response(events)["checks"].values())


def test_network_probe_refuses_reachable_internet(monkeypatch):
    from scripts import offline_check
    from contextlib import nullcontext

    monkeypatch.setattr(offline_check.socket, "create_connection", lambda *a, **kw: nullcontext())
    assert offline_check.check_network() == 2


def test_network_probe_allows_disconnected_machine(monkeypatch):
    from scripts import offline_check

    def disconnected(*args, **kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(offline_check.socket, "create_connection", disconnected)
    assert offline_check.check_network() == 0
