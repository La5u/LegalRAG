"""Check the sample contract through the real /ask NDJSON response."""
import argparse
import json
import os
import socket
import time

import httpx


QUESTION = (
    "샘플 소프트웨어 개발 용역 계약서에서 납품이 늦어지면 지체상금은 "
    "하루에 얼마이고 최대 한도는 얼마인가요?"
)
SAMPLE = "샘플_소프트웨어개발용역계약서"


def evaluate_response(events: list[dict]) -> dict:
    sources = [
        source
        for event in events if event.get("type") == "sources"
        for source in event.get("sources", [])
    ]
    answer = "".join(
        event.get("text", "") for event in events if event.get("type") == "token"
    )
    sample_sources = [
        source for source in sources if SAMPLE in source.get("raw_document", "")
    ]
    clauses = [
        source.get("text", "") for source in sample_sources
        if "제9조" in source.get("text", "")
    ]
    checks = {
        "Sample file cited": bool(sample_sources),
        "Sample evidence contains 제9조": bool(clauses),
        "Daily rate: 1천분의 2.5 / 0.25%": any(
            value in answer for value in ("1천분의 2.5", "0.25%")
        ),
        "Cap: 100분의 8 / 8%": any(
            value in answer for value in ("100분의 8", "8%")
        ),
        "Stream completed without errors": (
            any(event.get("type") == "done" for event in events)
            and not any(event.get("type") == "error" for event in events)
        ),
    }
    return {"checks": checks, "answer": answer, "clauses": clauses}


def check_network() -> int:
    for host in ("huggingface.co", "1.1.1.1"):
        try:
            with socket.create_connection((host, 443), timeout=2):
                print(
                    f"Internet is reachable ({host}:443). Disconnect networking "
                    "and retry, or use --allow-network for a dry run."
                )
                return 2
        except OSError:
            pass
    return 0


def run_check(server_pid: int) -> int:
    headers = {}
    if os.getenv("LEGALRAG_PROXY_SECRET"):
        headers["X-Internal-Proxy-Secret"] = os.environ["LEGALRAG_PROXY_SECRET"]
    # Ignore ambient HTTP proxies: the trust test talks only to loopback.
    with httpx.Client(
        base_url="http://127.0.0.1:8000", headers=headers,
        trust_env=False, timeout=600,
    ) as client:
        for attempt in range(120):
            os.kill(server_pid, 0)
            try:
                response = client.get("/config", timeout=1)
                response.raise_for_status()
                if response.json().get("allow_hosted_models") is not False:
                    raise RuntimeError("Hosted models must be disabled.")
                break
            except httpx.TransportError:
                if attempt == 119:
                    raise RuntimeError("Server did not start; see /tmp/legalrag-offline-web.log")
                time.sleep(1)
        events = []
        with client.stream("POST", "/ask", json={
            "question": QUESTION, "strategy": "graph", "generation_model": "local",
        }) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    events.append(json.loads(line))
    result = evaluate_response(events)
    print("Check | Result\n--- | ---")
    for label, passed in result["checks"].items():
        print(f"{label} | {'PASS' if passed else 'FAIL'}")
    print("\nAnswer:\n" + result["answer"])
    print("\nCited sample clause text (compare with 제9조 in the sample file):")
    print("\n\n".join(result["clauses"]) or "(No matching clause retrieved.)")
    passed = all(result["checks"].values())
    print("\n" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    if args.check_network:
        return check_network()
    if args.server_pid is None:
        parser.error("--server-pid is required")
    try:
        return run_check(args.server_pid)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
