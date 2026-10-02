"""Disable MuJoCo's bundled plugins that this project does not need.

    python scripts/prune_mujoco_plugins.py            # check, change nothing
    python scripts/prune_mujoco_plugins.py --apply
    python scripts/prune_mujoco_plugins.py --restore

`mujoco/__init__.py` loads every library in its `plugin/` directory at import
and does not catch failures, so one plugin that will not initialise makes
`import mujoco` fail outright, before any of this project's code runs. It
cannot be worked around in the environment; it has to be fixed in the install.

That happens on older CPUs. Measured on an Intel Xeon E5-2650 v3 (Haswell,
2014) under Windows 10 22H2: all eight MSVC runtime DLLs load, numpy and its
own native libraries load, mandatory ASLR is off, the files are not truncated
-- and the plugins still fail with

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed

which is what Windows reports when a library loads and its own initialisation
code then fails. The same wheels work on a 2025 CPU. torch's `c10.dll` fails
the same way on that machine, and the fix for that one is a version pin
(`requirements-haswell.txt`); this script is the other half.

**Only two of the six are needed.** `obj_decoder` and `stl_decoder` read the
MyoSuite meshes -- without them `spec.compile()` fails with "no decoder found
for mesh file ... sacrum.stl". The model uses no deformables, no
signed-distance geometry and no plugin actuators or sensors, so `elasticity`,
`sdf_plugin`, `actuator` and `sensor` can go. Verified by removing them: all
118 tests still pass.

**Which plugin is broken is decided by importing mujoco, not by loading each
library on its own.** A plugin links against the core `mujoco` library, which
`import mujoco` has already brought into the process by the time it loads them;
dlopening one directly fails for want of that dependency even on a machine
where everything works. So this keeps the two decoders, imports mujoco in a
subprocess to see whether that was enough, and puts everything back if it was
not.

Nothing is deleted -- disabled plugins move to `plugin_disabled/` beside the
originals, and `--restore` returns them.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

# The mesh decoders, by stem so the Linux and macOS builds match too.
REQUIRED = ("obj_decoder", "stl_decoder")
SUFFIXES = (".dll", ".so", ".dylib")

# Imports mujoco and builds the model, which is what the plugins are needed
# for. Run in a subprocess: a failing plugin cannot be unloaded once tried.
PROBE = """
import mujoco
print("mujoco", mujoco.__version__)
"""


def mujoco_plugin_dir() -> Path:
    """Where mujoco keeps its bundled plugins, found without importing it.

    `find_spec` does not execute the module, which matters here: importing
    mujoco is the thing that fails.
    """
    spec = importlib.util.find_spec("mujoco")
    if spec is None or not spec.origin:
        raise SystemExit("mujoco is not installed in %s" % sys.executable)
    return Path(spec.origin).parent / "plugin"


def libraries(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def mujoco_imports() -> tuple[bool, str]:
    """Whether `import mujoco` succeeds right now, and what it said."""
    done = subprocess.run([sys.executable, "-c", PROBE],
                          capture_output=True, text=True)
    out = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode == 0, out[-1] if out else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="disable the plugins this project does not need")
    ap.add_argument("--restore", action="store_true",
                    help="put previously disabled plugins back")
    args = ap.parse_args(argv)

    directory = mujoco_plugin_dir()
    parked = directory.parent / "plugin_disabled"
    print("plugins: %s" % directory)

    if args.restore:
        moved = [p for p in libraries(parked)]
        for path in moved:
            path.replace(directory / path.name)
            print("  restored %s" % path.name)
        if parked.is_dir() and not any(parked.iterdir()):
            parked.rmdir()
        ok, message = mujoco_imports()
        print("restored %d; import mujoco %s (%s)"
              % (len(moved), "works" if ok else "FAILS", message))
        return 0 if ok else 1

    present = libraries(directory)
    for path in present:
        print("  %-18s %s" % (path.name, "required (mesh decoder)"
                              if path.stem in REQUIRED else "not needed here"))

    ok, message = mujoco_imports()
    if ok:
        print("\nimport mujoco works (%s); nothing to do" % message)
        return 0

    print("\nimport mujoco FAILS: %s" % message)
    spare = [p for p in present if p.stem not in REQUIRED]
    if not spare:
        print("No plugin can be spared -- only the mesh decoders are left, and")
        print("the MyoSuite meshes cannot be read without them.")
        return 2
    if not args.apply:
        print("re-run with --apply to move %d plugin(s) to %s: %s"
              % (len(spare), parked.name, ", ".join(p.name for p in spare)))
        return 1

    parked.mkdir(exist_ok=True)
    for path in spare:
        path.replace(parked / path.name)
        print("  disabled %s" % path.name)

    ok, message = mujoco_imports()
    if ok:
        print("\nimport mujoco works now (%s)" % message)
        return 0

    # The decoders are the broken ones. Leaving them out would break the model
    # in a less obvious way later, so hand back the install as it was found.
    for path in libraries(parked):
        path.replace(directory / path.name)
    if not any(parked.iterdir()):
        parked.rmdir()
    print("\nstill FAILS: %s" % message)
    print("The mesh decoders themselves will not load, and the model cannot be")
    print("built without them. Everything has been put back.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
