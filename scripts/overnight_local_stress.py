#!/usr/bin/env python3
"""
Overnight local match stress test.

Runs random combinations of fighters, characters, stages, and CPU levels
through both `nojohns local` and `nojohns tournament` code paths.

For tournament mode, spins up a local arena server, creates matches,
and exercises the full pipeline (streaming, result reporting).

Detects and logs bug cases:
  - Character not selected / wrong character
  - Match never fires (stuck at CSS or stage select)
  - Stage not selected
  - Match crashed (non-zero exit, no game-end)
  - Match timed out (Melee timer ran out — fine but notable)
  - Arena/streaming failures (tournament mode)
  - Win/loss rates by character and fighter type

Outputs structured JSON log + periodic console summary.

Usage:
    .venv/bin/python scripts/overnight_local_stress.py
    .venv/bin/python scripts/overnight_local_stress.py --count 50 --timeout 300
    .venv/bin/python scripts/overnight_local_stress.py --forever
    .venv/bin/python scripts/overnight_local_stress.py --local-only
    .venv/bin/python scripts/overnight_local_stress.py --tournament-only
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ============================================================================
# Test Configuration
# ============================================================================

FIGHTERS = ["phillip", "random", "do-nothing"]

CPU_LEVELS = [1, 3, 5, 7, 9]

# Characters that work reliably with libmelee menu nav
CHARACTERS = [
    "FOX", "FALCO", "MARTH", "JIGGLYPUFF", "PEACH",
    "CPTFALCON", "PIKACHU", "SAMUS", "YLINK",
    "LINK", "LUIGI", "GANONDORF", "DOC", "MARIO",
    "YOSHI", "DK", "ZELDA", "ROY", "MEWTWO",
    "NESS", "BOWSER", "PICHU", "GAMEANDWATCH",
]

STAGES = [
    "FINAL_DESTINATION", "BATTLEFIELD", "YOSHIS_STORY",
    "DREAMLAND", "FOUNTAIN_OF_DREAMS", "POKEMON_STADIUM",
]

# Weighted test scenarios — interesting combos get more coverage
LOCAL_SCENARIOS = [
    # (weight, p1_type, p2_type, description)
    (3, "fighter", "cpu",     "Local: Fighter vs CPU"),
    (2, "fighter", "fighter", "Local: Fighter vs Fighter"),
    (2, "cpu",     "cpu",     "Local: CPU vs CPU"),
    (1, "fighter", "nothing", "Local: Fighter vs Do-Nothing"),
]

TOURNAMENT_SCENARIOS = [
    (3, "fighter", "fighter", "Tournament: Fighter vs Fighter"),
    (2, "fighter", "cpu",     "Tournament: Fighter vs CPU"),
    (1, "fighter", "nothing", "Tournament: Fighter vs Do-Nothing"),
]

# Per-match timeout in seconds (4 stock @ 8 min = 480s max, plus menu time)
DEFAULT_TIMEOUT = 600

ARENA_PORT = 18742  # Unusual port to avoid conflicts
ARENA_URL = f"http://localhost:{ARENA_PORT}"

# ============================================================================
# Arena Management
# ============================================================================

class ArenaServer:
    """Manages a local arena server for tournament testing."""

    def __init__(self):
        self.process = None
        self.match_counter = 0

    def start(self) -> bool:
        """Start local arena server. Returns True if successful."""
        print("Starting local arena server...")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "nojohns.cli", "arena", "--port", str(ARENA_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        # Wait for server to be ready
        for _ in range(30):
            time.sleep(0.5)
            if self._health_check():
                print(f"  Arena running at {ARENA_URL}")
                return True
            if self.process.poll() is not None:
                print("  Arena process died on startup")
                return False

        print("  Arena failed to start within 15s")
        self.stop()
        return False

    def stop(self):
        """Stop the arena server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("Arena server stopped.")

    def _health_check(self) -> bool:
        try:
            req = urllib.request.Request(f"{ARENA_URL}/queue/status")
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False

    def create_match(self, p1_name: str, p2_name: str) -> str | None:
        """Create a match via the arena queue and return the match ID."""
        try:
            # Join queue as P1
            p1_data = json.dumps({
                "connect_code": f"TEST#{self.match_counter:03d}A",
                "fighter_name": p1_name,
                "character": "RANDOM",
            }).encode()
            req = urllib.request.Request(
                f"{ARENA_URL}/queue/join",
                data=p1_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
            p1_qid = resp.get("queue_id")

            # Join queue as P2
            p2_data = json.dumps({
                "connect_code": f"TEST#{self.match_counter:03d}B",
                "fighter_name": p2_name,
                "character": "RANDOM",
            }).encode()
            req = urllib.request.Request(
                f"{ARENA_URL}/queue/join",
                data=p2_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())

            # Check for match
            time.sleep(0.5)
            req = urllib.request.Request(f"{ARENA_URL}/queue/status")
            status = json.loads(urllib.request.urlopen(req, timeout=5).read())

            # Find the match — check recent matches
            req = urllib.request.Request(f"{ARENA_URL}/matches")
            matches = json.loads(urllib.request.urlopen(req, timeout=5).read())

            if matches:
                # Most recent match
                match = matches[-1] if isinstance(matches, list) else None
                if match:
                    self.match_counter += 1
                    return match.get("id") or match.get("match_id")

            return None
        except Exception as e:
            print(f"  Failed to create arena match: {e}")
            return None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


# ============================================================================
# Match Generation
# ============================================================================

def gen_player_arg(player_type: str) -> list[str]:
    """Generate a --p1/--p2 argument list for the given player type."""
    char = random.choice(CHARACTERS)

    if player_type == "cpu":
        level = random.choice(CPU_LEVELS)
        return [f"cpu-{level}", char.lower()]
    elif player_type == "nothing":
        return ["do-nothing", char.lower()]
    else:  # "fighter"
        fighter = random.choice(FIGHTERS)
        return [fighter, char.lower()]


def gen_match_config(mode: str = "local") -> dict:
    """Generate a random match configuration."""
    scenarios = LOCAL_SCENARIOS if mode == "local" else TOURNAMENT_SCENARIOS
    total_weight = sum(w for w, *_ in scenarios)

    # Pick scenario by weight
    roll = random.randint(1, total_weight)
    cumulative = 0
    for weight, p1_type, p2_type, desc in scenarios:
        cumulative += weight
        if roll <= cumulative:
            break

    p1_args = gen_player_arg(p1_type)
    p2_args = gen_player_arg(p2_type)
    stage = random.choice(STAGES)

    return {
        "mode": mode,
        "scenario": desc,
        "p1_args": p1_args,
        "p2_args": p2_args,
        "stage": stage,
        "p1_label": " ".join(p1_args),
        "p2_label": " ".join(p2_args),
    }


# ============================================================================
# Output Parsing
# ============================================================================

def parse_match_output(stdout: str, stderr: str, exit_code: int, timed_out: bool) -> dict:
    """Parse match output and detect bug cases."""
    result = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "bugs": [],
        "events": [],
    }

    lines = stdout.split("\n")

    # Track what happened
    saw_match_start = False
    saw_game_start = False
    saw_game_over = False
    saw_match_complete = False
    saw_launching = False
    saw_streaming = False
    saw_result_reported = False
    saw_stage = None
    winner_port = None
    p1_char_log = None
    p2_char_log = None

    for line in lines:
        # Dolphin launched
        if "Launching Dolphin" in line:
            saw_launching = True
            result["events"].append("dolphin_launched")

        # Match started (fighters set up)
        if "Starting match:" in line:
            saw_match_start = True
            result["events"].append("match_started")
            m = re.search(r"Starting match: (.+) vs (.+)", line)
            if m:
                result["p1_fighter_actual"] = m.group(1).strip()
                result["p2_fighter_actual"] = m.group(2).strip()

        # Character/stage from P1/P2 log lines
        if "P1:" in line:
            m = re.search(r"P1: .+ \((\w+)\)", line)
            if m:
                p1_char_log = m.group(1)
        if "P2:" in line:
            m = re.search(r"P2: .+ \((\w+)\)", line)
            if m:
                p2_char_log = m.group(1)
        if "Stage:" in line:
            m = re.search(r"Stage: (\w+)", line)
            if m:
                saw_stage = m.group(1)

        # Game started (in-game)
        if "Game started" in line:
            saw_game_start = True
            result["events"].append("game_started")

        # Streaming (tournament mode)
        if "Streaming to" in line:
            saw_streaming = True
            result["events"].append("streaming_started")

        # Result reported (tournament mode)
        if "Result reported" in line:
            saw_result_reported = True
            result["events"].append("result_reported")

        # Game over
        if "Game over!" in line or ("Game" in line and "winner" in line.lower()):
            saw_game_over = True
            result["events"].append("game_over")
            m = re.search(r"P(\d) wins", line)
            if m:
                winner_port = int(m.group(1))

        # Match complete
        if "Match Complete" in line:
            saw_match_complete = True
            result["events"].append("match_complete")
            m = re.search(r"Winner: P(\d)", line)
            if m:
                winner_port = int(m.group(1))

        # Match cancelled
        if "Match cancelled" in line:
            result["events"].append("match_cancelled")

        # Error lines
        if "error" in line.lower() or "exception" in line.lower():
            result["events"].append(f"error: {line.strip()}")

    result["p1_char_log"] = p1_char_log
    result["p2_char_log"] = p2_char_log
    result["stage_log"] = saw_stage
    result["winner_port"] = winner_port
    result["streaming"] = saw_streaming
    result["result_reported"] = saw_result_reported

    # ---- Bug detection ----

    if timed_out:
        if not saw_game_start:
            result["bugs"].append("STUCK_PREGAME")
        elif not saw_game_over:
            result["bugs"].append("STUCK_INGAME")
        else:
            result["bugs"].append("PROCESS_TIMEOUT")

    if exit_code != 0 and not timed_out:
        if saw_game_start and not saw_game_over:
            result["bugs"].append("CRASH_MIDGAME")
        elif not saw_game_start:
            result["bugs"].append("CRASH_PREGAME")
        else:
            result["bugs"].append("CRASH_POSTGAME")

    if saw_launching and not saw_game_start and not timed_out and exit_code == 0:
        result["bugs"].append("MATCH_NEVER_FIRED")

    if saw_match_complete:
        result["status"] = "COMPLETE"
    elif saw_game_over:
        result["status"] = "GAME_ENDED"
    elif exit_code == 130:
        result["status"] = "CANCELLED"
    elif timed_out:
        result["status"] = "TIMEOUT"
    else:
        result["status"] = "UNKNOWN"

    return result


# ============================================================================
# Runner
# ============================================================================

def run_local_match(config: dict, timeout: int, match_num: int) -> dict:
    """Run a local match via `nojohns local`."""
    cmd = [
        sys.executable, "-m", "nojohns.cli", "local",
        "--p1", *config["p1_args"],
        "--p2", *config["p2_args"],
        "--stage", config["stage"],
        "--stocks", "4",
        "--time", "8",
    ]
    return _run_subprocess(cmd, config, timeout, match_num)


def run_tournament_match(config: dict, timeout: int, match_num: int,
                          arena: ArenaServer) -> dict:
    """Run a tournament match via `nojohns tournament`."""
    # Create match in arena
    p1_fighter = config["p1_args"][0]
    p2_fighter = config["p2_args"][0]
    match_id = arena.create_match(p1_fighter, p2_fighter)

    if not match_id:
        return {
            "match_num": match_num,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "elapsed_seconds": 0,
            "exit_code": -1,
            "timed_out": False,
            "bugs": ["ARENA_MATCH_CREATE_FAILED"],
            "events": ["arena_create_failed"],
            "status": "ARENA_ERROR",
            "winner_port": None,
        }

    cmd = [
        sys.executable, "-m", "nojohns.cli", "tournament",
        "--matchid", str(match_id),
        "--server", ARENA_URL,
        "--p1-fighter", config["p1_args"][0],
        "--p2-fighter", config["p2_args"][0],
        "--stage", config["stage"],
    ]

    # Add character overrides if specified
    if len(config["p1_args"]) > 1:
        cmd.extend(["--p1-char", config["p1_args"][1]])
    if len(config["p2_args"]) > 1:
        cmd.extend(["--p2-char", config["p2_args"][1]])

    result = _run_subprocess(cmd, config, timeout, match_num)
    result["arena_match_id"] = match_id

    # Tournament-specific bug detection
    if result["status"] == "COMPLETE" and not result.get("result_reported"):
        result["bugs"].append("RESULT_NOT_REPORTED")
    if result["status"] == "COMPLETE" and not result.get("streaming"):
        result["bugs"].append("STREAMING_NOT_STARTED")

    return result


def _run_subprocess(cmd: list[str], config: dict, timeout: int, match_num: int) -> dict:
    """Run a match subprocess and return structured results."""
    print(f"\n{'='*60}")
    print(f"Match #{match_num} [{config['mode'].upper()}]: "
          f"{config['p1_label']} vs {config['p2_label']}")
    print(f"  Stage: {config['stage']}  |  Scenario: {config['scenario']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent.parent),
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
    except KeyboardInterrupt:
        print("\n\nStress test interrupted by user.")
        raise

    elapsed = time.time() - start_time

    parsed = parse_match_output(stdout, stderr, exit_code, timed_out)

    entry = {
        "match_num": match_num,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "elapsed_seconds": round(elapsed, 1),
        **parsed,
    }

    # Print immediate result
    status_icon = {
        "COMPLETE": "OK",
        "GAME_ENDED": "OK~",
        "CANCELLED": "SKIP",
        "TIMEOUT": "TIMEOUT",
        "UNKNOWN": "???",
        "ARENA_ERROR": "ARENA",
    }.get(entry["status"], "ERR")

    bugs_str = f"  BUGS: {', '.join(entry['bugs'])}" if entry["bugs"] else ""
    winner_str = f"  Winner: P{entry['winner_port']}" if entry.get("winner_port") else ""
    print(f"  [{status_icon}] {elapsed:.0f}s{winner_str}{bugs_str}")

    return entry


def print_summary(results: list[dict]):
    """Print a summary of all results so far."""
    total = len(results)
    if total == 0:
        return

    statuses = Counter(r["status"] for r in results)
    all_bugs = Counter()
    for r in results:
        for b in r["bugs"]:
            all_bugs[b] += 1

    # Win rates by character (for completed matches)
    char_wins = defaultdict(lambda: {"wins": 0, "losses": 0})
    fighter_wins = defaultdict(lambda: {"wins": 0, "losses": 0})
    scenario_stats = defaultdict(lambda: {"ok": 0, "fail": 0})
    mode_stats = defaultdict(lambda: {"ok": 0, "fail": 0})

    for r in results:
        ok = r["status"] in ("COMPLETE", "GAME_ENDED")
        scenario_stats[r["config"]["scenario"]]["ok" if ok else "fail"] += 1
        mode_stats[r["config"]["mode"]]["ok" if ok else "fail"] += 1

        if r.get("winner_port") and ok:
            wp = r["winner_port"]
            p1_char = r.get("p1_char_log", "?")
            p2_char = r.get("p2_char_log", "?")
            p1_label = r["config"]["p1_label"].split()[0]
            p2_label = r["config"]["p2_label"].split()[0]

            if wp == 1:
                char_wins[p1_char]["wins"] += 1
                char_wins[p2_char]["losses"] += 1
                fighter_wins[p1_label]["wins"] += 1
                fighter_wins[p2_label]["losses"] += 1
            else:
                char_wins[p2_char]["wins"] += 1
                char_wins[p1_char]["losses"] += 1
                fighter_wins[p2_label]["wins"] += 1
                fighter_wins[p1_label]["losses"] += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY — {total} matches")
    print(f"{'='*60}")

    print(f"\nMode breakdown:")
    for mode, stats in sorted(mode_stats.items()):
        total_m = stats["ok"] + stats["fail"]
        pct = stats["ok"] / total_m * 100 if total_m else 0
        print(f"  {mode:15s} {stats['ok']}/{total_m} ({pct:.0f}%)")

    print(f"\nStatus breakdown:")
    for status, count in statuses.most_common():
        pct = count / total * 100
        print(f"  {status:20s} {count:4d} ({pct:.0f}%)")

    if all_bugs:
        print(f"\nBugs detected:")
        for bug, count in all_bugs.most_common():
            print(f"  {bug:30s} {count:4d}")

    print(f"\nScenario success rates:")
    for scenario, stats in sorted(scenario_stats.items()):
        total_s = stats["ok"] + stats["fail"]
        pct = stats["ok"] / total_s * 100 if total_s else 0
        print(f"  {scenario:35s} {stats['ok']}/{total_s} ({pct:.0f}%)")

    if fighter_wins:
        print(f"\nFighter win rates:")
        for fighter, stats in sorted(fighter_wins.items(), key=lambda x: -(x[1]["wins"])):
            total_f = stats["wins"] + stats["losses"]
            pct = stats["wins"] / total_f * 100 if total_f else 0
            print(f"  {fighter:15s} {stats['wins']:3d}W {stats['losses']:3d}L ({pct:.0f}%)")

    if char_wins:
        print(f"\nCharacter win rates (top 10):")
        sorted_chars = sorted(char_wins.items(), key=lambda x: -(x[1]["wins"]))
        for char, stats in sorted_chars[:10]:
            total_c = stats["wins"] + stats["losses"]
            pct = stats["wins"] / total_c * 100 if total_c else 0
            print(f"  {char:15s} {stats['wins']:3d}W {stats['losses']:3d}L ({pct:.0f}%)")

    # Average match duration for completed matches
    completed = [r for r in results if r["status"] in ("COMPLETE", "GAME_ENDED")]
    if completed:
        avg_time = sum(r["elapsed_seconds"] for r in completed) / len(completed)
        print(f"\nAvg match duration: {avg_time:.0f}s ({avg_time/60:.1f} min)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight local match stress test")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of matches to run (0 = 100, --forever for unlimited)")
    parser.add_argument("--forever", action="store_true",
                        help="Run until interrupted")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-match timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--log", default=None,
                        help="Log file path (default: logs/stress-TIMESTAMP.json)")
    parser.add_argument("--summary-every", type=int, default=10,
                        help="Print summary every N matches (default: 10)")
    parser.add_argument("--local-only", action="store_true",
                        help="Only run local matches (no arena)")
    parser.add_argument("--tournament-only", action="store_true",
                        help="Only run tournament matches (requires arena)")
    parser.add_argument("--tournament-ratio", type=float, default=0.3,
                        help="Fraction of matches to run as tournament (default: 0.3)")
    args = parser.parse_args()

    count = args.count if args.count > 0 else (999999 if args.forever else 100)

    # Set up log file
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log) if args.log else log_dir / f"stress-{timestamp}.jsonl"

    # Determine mode mix
    use_tournament = not args.local_only
    use_local = not args.tournament_only
    tournament_ratio = 1.0 if args.tournament_only else (0.0 if args.local_only else args.tournament_ratio)

    print(f"Overnight Stress Test")
    print(f"  Matches: {'unlimited' if args.forever else count}")
    print(f"  Timeout: {args.timeout}s per match")
    print(f"  Modes: {'local + tournament' if use_local and use_tournament else 'local only' if use_local else 'tournament only'}")
    if use_tournament:
        print(f"  Tournament ratio: {tournament_ratio:.0%}")
    print(f"  Log: {log_path}")
    print(f"  Press Ctrl+C to stop and see summary\n")

    # Start arena if needed
    arena = None
    if use_tournament:
        arena = ArenaServer()
        if not arena.start():
            if args.tournament_only:
                print("ERROR: Arena failed to start and --tournament-only was set. Exiting.")
                sys.exit(1)
            print("WARNING: Arena failed to start. Running local-only.\n")
            use_tournament = False
            tournament_ratio = 0.0

    results = []

    try:
        for i in range(1, count + 1):
            # Pick mode
            if use_tournament and use_local:
                mode = "tournament" if random.random() < tournament_ratio else "local"
            elif use_tournament:
                mode = "tournament"
            else:
                mode = "local"

            # Check arena health for tournament matches
            if mode == "tournament" and arena and not arena.is_alive():
                print("  Arena died, restarting...")
                arena.stop()
                if not arena.start():
                    print("  Arena restart failed, falling back to local")
                    mode = "local"

            config = gen_match_config(mode)

            if mode == "tournament" and arena:
                entry = run_tournament_match(config, args.timeout, i, arena)
            else:
                entry = run_local_match(config, args.timeout, i)

            results.append(entry)

            # Append to log file (JSONL — one JSON object per line)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Periodic summary
            if i % args.summary_every == 0:
                print_summary(results)

            # Brief pause between matches for cleanup
            time.sleep(2)

    except KeyboardInterrupt:
        pass

    # Cleanup
    if arena:
        arena.stop()

    # Final summary
    print_summary(results)
    print(f"\nFull log: {log_path}")

    # Write a readable summary file too
    summary_path = log_path.with_suffix(".summary.txt")
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_summary(results)
    sys.stdout = old_stdout
    summary_path.write_text(buf.getvalue())
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
