"""Find a MuJoCo build whose model compiler works on this machine.

    python scripts/find_working_mujoco.py
    python scripts/find_working_mujoco.py --versions 3.3.0 3.1.6 2.3.7
    python scripts/find_working_mujoco.py --dlls

On one 2014 Xeon, MuJoCo 3.3 imports and then dies with an access violation
(0xC0000005) compiling a single-sphere model -- under uv's interpreter and
under python.org's, with PATH stripped to Windows' own directories, with no
conflicting MuJoCo DLL anywhere on PATH, on a supported OS, with every MSVC
runtime DLL loading and numpy's native libraries working. So the fault is in
MuJoCo's own compiled code on that CPU.

Only one variable was never swept: the MuJoCo version. 3.6 and 3.2.7 never
reached the compiler there, because both died earlier loading their bundled
plugins. This builds one throwaway virtual environment per version, installs
**only** mujoco into it, moves the whole plugin directory aside -- a
single-sphere model needs no mesh decoder, so nothing is lost -- and compiles.
Each attempt runs in its own subprocess, so an access violation reports as an
exit code instead of taking this script with it.

What to do with the answer:

* A version that compiles is a candidate. Check it against this project with
  `scripts/diagnose_mujoco.py`, because the MjSpec API this env uses to
  restructure the model only exists from 3.1 and changed in 3.3 and 3.6, and
  myosuite pins mujoco tightly (2.12.2 wants >=3.6,<3.7; 2.11.6 wants 3.3).
* No version compiling means MuJoCo cannot run on the machine at all, and the
  causes left -- endpoint-security injection, marginal memory -- are outside
  this repository. `--dlls` helps with the first of those: it lists the
  libraries loaded into the process that belong to neither Windows nor the
  environment, which is where an injected hook shows up.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Newest first. 3.6 is what requirements.txt pins, 3.3 what the Haswell pins
# use; below that the MjSpec API this project needs starts to disappear, but
# knowing where the compiler starts working is worth more than convenience.
VERSIONS = ("3.6.0", "3.3.0", "3.2.7", "3.1.6", "3.0.1", "2.3.7")

COMPILE = r"""
import importlib.util, json, os, pathlib, sys

# Move the bundled plugins aside before importing: on some machines they fail
# to initialise and mujoco/__init__ does not catch it. A single-sphere model
# needs no mesh decoder, so this costs nothing and isolates the compiler.
spec = importlib.util.find_spec("mujoco")
plugin = pathlib.Path(spec.origin).parent / "plugin"
if plugin.is_dir():
    plugin.rename(plugin.with_name("plugin_off"))

import mujoco
out = {"version": mujoco.__version__}
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><body><geom size='.1'/><joint type='free'/>"
    "</body></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
out["nq"] = m.nq
print("RESULT " + json.dumps(out))
"""

LIST_DLLS = r"""
import ctypes, ctypes.wintypes as w, json, os, sys, pathlib
try:
    import mujoco  # noqa: F401  -- load it, so its dependencies are in too
except Exception as exc:
    print("import failed: %r" % exc)

psapi = ctypes.WinDLL("psapi", use_last_error=True)
handle = ctypes.windll.kernel32.GetCurrentProcess()
needed = w.DWORD()
buf = (ctypes.c_void_p * 2048)()
psapi.EnumProcessModules(handle, ctypes.byref(buf), ctypes.sizeof(buf),
                         ctypes.byref(needed))
count = min(needed.value // ctypes.sizeof(ctypes.c_void_p), 2048)

name = ctypes.create_unicode_buffer(1024)
root = (os.environ.get("SystemRoot") or "C:\\Windows").lower()
home = str(pathlib.Path(sys.executable).parents[1]).lower()
foreign = []
for i in range(count):
    psapi.GetModuleFileNameExW(handle, buf[i], name, 1024)
    path = name.value
    low = path.lower()
    if low.startswith(root) or low.startswith(home):
        continue
    foreign.append(path)
print("FOREIGN " + json.dumps(sorted(set(foreign))))
"""


def make_venv(directory: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", str(directory)],
                   check=True, capture_output=True)
    python = directory / "Scripts" / "python.exe"
    return python if python.exists() else directory / "bin" / "python"


def attempt(version: str, workdir: Path, quiet: bool):
    """Install this mujoco alone and compile a trivial model with it."""
    venv = workdir / ("mj-" + version.replace(".", "_"))
    try:
        python = make_venv(venv)
    except subprocess.CalledProcessError as exc:
        return "venv failed", (exc.stderr or b"").decode(errors="replace")[-200:]

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "mujoco==%s" % version],
        capture_output=True, text=True,
    )
    if install.returncode != 0:
        tail = [ln for ln in (install.stderr or "").splitlines() if ln.strip()]
        return "not installable", tail[-1] if tail else ""

    done = subprocess.run([str(python), "-c", COMPILE],
                          capture_output=True, text=True)
    for line in (done.stdout or "").splitlines():
        if line.startswith("RESULT "):
            return "COMPILES", json.loads(line[7:])
    if done.returncode and not 0 < done.returncode < 128:
        return "native crash", "exit %#x" % (done.returncode & 0xFFFFFFFF)
    tail = [ln for ln in (done.stdout + done.stderr).splitlines() if ln.strip()]
    return "failed", tail[-1] if tail else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--versions", nargs="+", default=list(VERSIONS))
    ap.add_argument("--dlls", action="store_true",
                    help="also list libraries loaded into the process that "
                         "belong to neither Windows nor this environment")
    ap.add_argument("--keep", action="store_true",
                    help="leave the throwaway environments in place")
    args = ap.parse_args(argv)

    print("interpreter %s" % sys.executable)
    print("python      %s\n" % sys.version.split()[0])

    workdir = Path(tempfile.mkdtemp(prefix="mujoco-sweep-"))
    working = []
    try:
        for version in args.versions:
            print("  mujoco %-7s " % version, end="", flush=True)
            verdict, detail = attempt(version, workdir, quiet=True)
            if verdict == "COMPILES":
                print("COMPILES      (nq %d)" % detail["nq"])
                working.append(version)
            else:
                print("%-13s %s" % (verdict, detail))
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    print()
    if working:
        print("compiles on this machine: %s" % ", ".join(working))
        print("Check one against this project:  "
              "python scripts/diagnose_mujoco.py")
    else:
        print("no tested MuJoCo build compiles a model on this machine.")

    if args.dlls:
        print("\nlibraries loaded from neither Windows nor this environment:")
        done = subprocess.run([sys.executable, "-c", LIST_DLLS],
                              capture_output=True, text=True)
        shown = False
        for line in (done.stdout or "").splitlines():
            if line.startswith("FOREIGN "):
                for path in json.loads(line[8:]):
                    print("  %s" % path)
                    shown = True
        if not shown:
            print("  (none -- nothing is being injected into the process)")

    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
