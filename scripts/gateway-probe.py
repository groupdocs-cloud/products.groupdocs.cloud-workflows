#!/usr/bin/env python3
"""Does the gateway serve one user's requests concurrently, or queue them?

Sends a warmup, then 3 sequential requests, then 6 concurrent ones, all with
unique prompts (no cache reuse). If the gateway is parallel, each concurrent
request takes about one service time and the batch wall equals one service
time. If it queues per user, the sorted concurrent durations form a staircase
(1x, 2x, 3x ... service time) and the wall equals the sequential total.

Reads OPENAI_BASE_URL, OPENAI_API_KEY, TRANSLATION_MODEL from the environment.
Prints timings and a verdict; never prints the key.
"""

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ.get("TRANSLATION_MODEL", "professionalize-gpt-oss")
URL = BASE + "/chat/completions"
NONCE = hex(int(time.time()))[2:]
TIMEOUT = 170

CTX = ssl.create_default_context()
CA = "/root/.ccr/ca-bundle.crt"


def call(tag, minimal=False):
    if minimal:
        prompt = f"Reply with only: ok ({tag}-{NONCE})"
    else:
        prompt = (
            "Translate to Spanish, reply with the translation only: "
            f"'The meeting starts at seven and the report is due tomorrow ({tag}-{NONCE}).'"
        )
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "User-Agent": "gateway-probe/1.0",
        },
    )
    global CTX
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as resp:
            payload = json.loads(resp.read())
            status = resp.status
    except ssl.SSLCertVerificationError:
        if not os.path.exists(CA):
            raise
        CTX = ssl.create_default_context(cafile=CA)
        return call(tag)
    except urllib.error.HTTPError as e:
        return {"tag": tag, "status": e.code, "dur": time.monotonic() - start, "chars": 0}
    except Exception as e:
        return {"tag": tag, "status": f"error:{type(e).__name__}", "dur": time.monotonic() - start, "chars": 0}
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return {"tag": tag, "status": status, "dur": time.monotonic() - start, "chars": len(text)}


def show(label, results, wall):
    busy = sum(r["dur"] for r in results)
    print(f"\n{label}: wall {wall:.1f}s, busy {busy:.1f}s")
    for r in sorted(results, key=lambda r: r["dur"]):
        print(f"  {r['tag']:>6}  {r['dur']:6.1f}s  status={r['status']}  reply_chars={r['chars']}")
    return busy


def ramp():
    """Climb 1,2,4,...,128 concurrent minimal requests until the gateway chokes."""
    host = URL.split("//", 1)[-1].split("/", 1)[0]
    print(f"gateway ramp: host={host} model={MODEL} nonce={NONCE}")
    w = call("warmup", minimal=True)
    print(f"warmup: {w['dur']:.1f}s status={w['status']}")
    if w["status"] != 200:
        print("warmup failed - aborting")
        sys.exit(1)

    base_median = None
    last_rate = 0.0
    flat = 0
    highest_clean = 1
    first_degraded = None
    print(f"\n{'N':>4} {'ok':>4} {'err':>4} {'wall':>7} {'median':>7} {'max':>7} {'req/s':>6} {'par':>6}")
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        time.sleep(3)
        t = time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(lambda i: call(f"r{n}.{i}", minimal=True), range(n)))
        wall = time.monotonic() - t
        durs = sorted(r["dur"] for r in results)
        ok = sum(1 for r in results if r["status"] == 200)
        err = n - ok
        median = durs[len(durs) // 2]
        busy = sum(durs)
        rate = ok / wall if wall else 0.0
        print(
            f"{n:>4} {ok:>4} {err:>4} {wall:>6.1f}s {median:>6.1f}s {durs[-1]:>6.1f}s {rate:>6.2f} {busy / wall:>6.2f}"
        )
        if base_median is None:
            base_median = median
        degraded = err > 0 or median > 1.5 * base_median
        if not degraded:
            highest_clean = n
        elif first_degraded is None:
            first_degraded = n
        if err > n // 4:
            print(f"\nstopping: {err} of {n} failed")
            break
        if rate < last_rate * 1.15:
            flat += 1
            if flat >= 2:
                print(f"\nstopping: throughput flat since N={n // 4}")
                break
        else:
            flat = 0
        last_rate = max(last_rate, rate)

    print(f"\nhighest clean level: {highest_clean} (no errors, median within 1.5x of unloaded)")
    if first_degraded:
        print(f"first degraded level: {first_degraded}")
    else:
        print("no degradation observed up to the last level tried")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "ramp":
        ramp()
        return
    host = URL.split("//", 1)[-1].split("/", 1)[0]
    print(f"gateway probe: host={host} model={MODEL} nonce={NONCE}")

    t = time.monotonic()
    w = call("warmup")
    print(f"\nwarmup: {w['dur']:.1f}s status={w['status']} reply_chars={w['chars']}")
    if not isinstance(w["status"], int) or w["status"] != 200:
        print("warmup failed - aborting")
        sys.exit(1)

    t = time.monotonic()
    seq = [call(f"seq{i}") for i in range(3)]
    seq_wall = time.monotonic() - t
    show("sequential x3", seq, seq_wall)
    seq_mean = sum(r["dur"] for r in seq) / len(seq)

    t = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        conc = list(pool.map(call, [f"conc{i}" for i in range(6)]))
    conc_wall = time.monotonic() - t
    busy = show("concurrent x6", conc, conc_wall)

    eff = busy / conc_wall if conc_wall else 0.0
    print(f"\nsequential mean service time: {seq_mean:.1f}s")
    print(f"effective parallelism: {eff:.2f} of 6")
    if eff >= 4.0:
        print("verdict: gateway serves this user's requests CONCURRENTLY")
    elif eff <= 1.5:
        print("verdict: gateway QUEUES this user's requests (serialized)")
    else:
        print(f"verdict: PARTIAL concurrency, roughly {round(eff)} slots for this user")


if __name__ == "__main__":
    main()
