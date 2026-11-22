#!/usr/bin/env python3
"""Run test data for the Stella IR optimiser.

Temp files live under <project_root>/build/. Because ./stella bind-mounts
$PWD → /work, all paths handed to ./stella are relative to project_root;
absolute host paths (outside the mount) would be invisible to the container.

Dependencies: Python 3.10+ stdlib only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class Runner:
    def __init__(self, project_root: Path, stella: str, stage2: str, build: Path):
        self.project_root = project_root.resolve()
        self.stella = stella
        self.stage2 = stage2
        self.build = build.resolve()
        self.build.mkdir(parents=True, exist_ok=True)

    # ---- paths

    def _rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.project_root))

    def _ensure_local(self, src: Path) -> Path:
        try:
            src.resolve().relative_to(self.project_root)
            return src
        except ValueError:
            dst = self.build / "external" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return dst

    # ---- subprocess

    def _stella(
        self, *args: str, log_dir: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        cmd = [self.stella, *args]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        stderr = _decode(result.stderr)
        stdout = _decode(result.stdout)
        log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"stella-{args[0]}.log"
            log_path.write_text(
                f"$ {' '.join(cmd)}\nexit {result.returncode}\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
            )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"`{self.stella} {' '.join(args)}` failed (exit {result.returncode})"
                + (f"\n  log: {log_path}" if log_path else "")
                + f"\n  stderr:\n{_indent(stderr.strip() or '<empty>')}"
            )
        return result

    def _stage2_pipe(self, mode: str, inp: Path, out: Path, log_dir: Path | None = None) -> None:
        cmd = [self.stage2, mode]
        with open(inp, "rb") as fin, open(out, "wb") as fout:
            result = subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
            )
        stderr = _decode(result.stderr)
        log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"stage2-{mode}.log"
            log_path.write_text(
                f"$ {' '.join(cmd)} < {self._rel(inp)}\nexit {result.returncode}\n"
                f"--- stderr ---\n{stderr}\n"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"`{self.stage2} {mode}` failed (exit {result.returncode})"
                + (f"\n  log: {log_path}" if log_path else "")
                + f"\n  stderr:\n{_indent(stderr.strip() or '<empty>')}"
            )

    # ---- pipeline steps

    def to_json_ir(
        self, src: Path, dst: Path, config_interface: Path | None, log_dir: Path | None = None
    ) -> Path | None:
        """Convert src → JSON IR at dst; return interface path or None."""
        ext = src.suffix.lower()
        if ext == ".json":
            shutil.copy2(src, dst)
            if config_interface is not None:
                iface = dst.with_name(dst.stem + ".interface.json")
                iface_src = self._ensure_local(config_interface)
                shutil.copy2(iface_src, iface)
                return iface
            return None
        if ext == ".sir":
            self._stella(
                "ir-convert", self._rel(src), "--to", "json", "-o", self._rel(dst), log_dir=log_dir
            )
            if config_interface is not None:
                iface = dst.with_name(dst.stem + ".interface.json")
                shutil.copy2(self._ensure_local(config_interface), iface)
                return iface
            return None
        if ext == ".stella":
            iface = dst.with_name(dst.stem + ".interface.json")
            self._stella(
                "compile-ir",
                self._rel(src),
                "--to",
                "json",
                "-o",
                self._rel(dst),
                "--interface",
                self._rel(iface),
                log_dir=log_dir,
            )
            return iface
        raise ValueError(f"Unknown source extension {ext!r} for {src}")

    def check_ir(self, ir: Path, log_dir: Path | None = None) -> None:
        self._stella("ir-check", self._rel(ir), log_dir=log_dir)

    def run_ir(
        self, ir: Path, request: Path, interface: Path | None, log_dir: Path | None = None
    ) -> dict:
        if interface is not None and interface.exists():
            result = self._stella(
                "ir-run-source",
                self._rel(ir),
                "--interface",
                self._rel(interface),
                "--request",
                self._rel(request),
                log_dir=log_dir,
            )
        else:
            result = self._stella(
                "ir-run",
                self._rel(ir),
                "--request",
                self._rel(request),
                log_dir=log_dir,
            )
        text = _decode(result.stdout).strip()
        if not text:
            raise RuntimeError(f"ir-run produced empty output for {ir.name}")
        return json.loads(text)

    def apply_pipeline(self, inp: Path, modes: list[str], stage_dir: Path, log_dir: Path) -> Path:
        current = inp
        for i, mode in enumerate(modes):
            out = stage_dir / f"pass-{i:02d}-{mode}.json"
            self._stage2_pipe(mode, current, out, log_dir=log_dir)
            current = out
        return current


# --------------------------------------------------------------- main


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def _compare(actual, expected) -> tuple[bool, str]:
    if actual == expected:
        return True, ""
    return False, (f"expected: {json.dumps(expected)}\n" f"actual:   {json.dumps(actual)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--stella", default="./stella")
    ap.add_argument("--stage2", default="./stage-2")
    ap.add_argument("--store-ir", default=None, metavar="DIR")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--stop-on-first-failure", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    if not project_root.is_dir():
        print(f"error: project root not found: {project_root}", file=sys.stderr)
        return 2

    build = Path(args.build_dir).resolve() if args.build_dir else project_root / "build"
    try:
        build.relative_to(project_root)
    except ValueError:
        print(f"error: build dir {build} must be inside project root", file=sys.stderr)
        return 2

    runner = Runner(project_root, args.stella, args.stage2, build)

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        config = json.load(f)
    base = config_path.parent

    store_dir = Path(args.store_ir).resolve() if args.store_ir else None
    if store_dir:
        store_dir.mkdir(parents=True, exist_ok=True)

    tests = config.get("tests", [])
    if not tests:
        print(f"no tests in {config_path}", file=sys.stderr)
        return 2

    total = 0
    passed = 0
    failures: list[tuple[str, str, str]] = []

    def fail(test_name: str, label: str, message: str) -> None:
        failures.append((test_name, label, message))
        print(f"  [FAIL] {label}")
        for line in message.splitlines():
            print(f"         {line}")
        if args.stop_on_first_failure:
            raise KeyboardInterrupt

    try:
        for test in tests:
            name = test["name"]
            source = runner._ensure_local((base / test["source"]).resolve())
            config_iface = (base / test["interface"]).resolve() if test.get("interface") else None
            cases = test.get("cases", [])
            passes = test.get("passes", [])

            print(f"\n=== {name} ===")
            print(f"    source:  {source}")
            print(f"    cases:   {len(cases)}")
            print(f"    passes:  {len(passes)}")

            if not source.exists():
                fail(name, "<source>", f"file not found: {source}")
                continue

            ts = time.strftime("%Y%m%d-%H%M%S")
            work = build / f"{_sanitize(name)}-{ts}"
            work.mkdir(parents=True, exist_ok=True)
            logs = work / "logs"
            logs.mkdir(exist_ok=True)

            try:
                input_ir = work / "input.json"
                try:
                    interface = runner.to_json_ir(source, input_ir, config_iface, log_dir=logs)
                except Exception as e:
                    fail(name, "<convert>", str(e))
                    continue

                # Request files
                requests: list[Path] = []
                for cidx, case in enumerate(cases):
                    req = work / f"case-{cidx}.request.json"
                    req.write_text(json.dumps(case["input"]))
                    requests.append(req)

                # Resolve expected responses
                expected: list[dict | None] = []
                for cidx, (case, req) in enumerate(zip(cases, requests)):
                    exp = case.get("expected")
                    if exp is not None:
                        expected.append(exp)
                        # Sanity check: the *input* IR must already agree with
                        # `expected`. If it doesn't, the test data is wrong (or
                        # the input itself is broken) — report it as a separate
                        # failure, but still run the pipelines below, since they
                        # may reveal a real bug on top of the broken data.
                        try:
                            baseline = runner.run_ir(input_ir, req, interface, log_dir=logs)
                        except Exception as e:
                            fail(
                                name,
                                f"case{cidx}/<input>",
                                "input IR failed to run; `expected` cannot be "
                                f"trusted for this case (test data is wrong): {e}",
                            )
                        else:
                            ok, diff = _compare(baseline, exp)
                            if ok:
                                if args.verbose:
                                    print(f"    case{cidx}: input result matches expected")
                            else:
                                fail(
                                    name,
                                    f"case{cidx}/<input>",
                                    "input IR disagrees with `expected` "
                                    "(test data is wrong or the input is broken):\n" + diff,
                                )
                        continue
                    try:
                        baseline = runner.run_ir(input_ir, req, interface, log_dir=logs)
                        expected.append(baseline)
                        if args.verbose:
                            print(f"    case{cidx}: baseline = {json.dumps(baseline)}")
                    except Exception as e:
                        expected.append(None)
                        fail(name, f"case{cidx}/<baseline>", str(e))

                # Run each pass pipeline
                for pidx, modes in enumerate(passes):
                    label = f"pass{pidx}:{'->'.join(modes) if modes else 'identity'}"
                    stage_dir = work / f"pass-{pidx:02d}"
                    stage_dir.mkdir(exist_ok=True)

                    try:
                        out_ir = runner.apply_pipeline(input_ir, modes, stage_dir, log_dir=logs)
                    except Exception as e:
                        fail(name, label, str(e))
                        continue

                    if store_dir:
                        dest = (
                            store_dir
                            / _sanitize(name)
                            / (f"pass{pidx}-{'-'.join(modes) if modes else 'identity'}.json")
                        )
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(out_ir, dest)
                        if args.verbose:
                            print(f"    [store] {dest}")

                    try:
                        runner.check_ir(out_ir, log_dir=logs)
                    except Exception as e:
                        fail(name, label, f"ir-check: {e}")
                        continue

                    for cidx, req in enumerate(requests):
                        case_label = f"{label}/case{cidx}"
                        try:
                            actual = runner.run_ir(out_ir, req, interface, log_dir=logs)
                        except Exception as e:
                            total += 1
                            fail(name, case_label, f"run: {e}")
                            continue

                        if expected[cidx] is None:
                            if args.verbose:
                                print(f"  [SKIP] {case_label}")
                            continue

                        total += 1
                        ok, diff = _compare(actual, expected[cidx])
                        if ok:
                            passed += 1
                            if args.verbose:
                                print(f"  [ OK ] {case_label}")
                        else:
                            fail(name, case_label, f"mismatch:\n{diff}")
            finally:
                if args.keep_temp:
                    print(f"    [temp] {work}")
                else:
                    shutil.rmtree(work, ignore_errors=True)

    except KeyboardInterrupt:
        print("\nAborted.")

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} checks passed, {len(failures)} failure(s)")
    if failures:
        print("\nFailures:")
        for n, l, msg in failures:
            first = msg.splitlines()[0] if msg else ""
            print(f"  - {n} [{l}]: {first}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
