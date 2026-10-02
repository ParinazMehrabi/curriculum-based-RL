"""Find out how much of MuJoCo works on this machine, one step at a time.

    python scripts/diagnose_mujoco.py

Each stage runs in its own subprocess, because the failures this is for are
native: an access violation takes the interpreter with it, and a plugin that
will not initialise cannot be unloaded and retried. A stage that crashes
therefore does not stop the ones after it.

Written for a 2014 Xeon where the usual checks all pass -- every MSVC runtime
DLL loads, numpy's own native libraries load, mandatory ASLR is off, the OS is
supported, the files are not truncated -- and MuJoCo still fails, first at
`WinError 1114` loading its bundled plugins and then, once those were pruned,
with an access violation inside `spec.compile()`. The question these stages
answer is which layer is actually broken:

* import           -- is the extension module usable at all
* trivial compile  -- is the model compiler usable
* mesh compile     -- is mesh processing usable (convex hulls, decoding)
* myosuite as-is   -- is MyoSuite's own model usable, unedited
* myosuite edited  -- and does this project's MjSpec restructuring work

The first stage that fails is the answer, and everything above it is fine.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

V5 = Path(__file__).resolve().parents[1]

TRIVIAL = """
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom size='.1'/>"
                                   "<joint type='free'/></body></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print("nq %d, stepped" % m.nq)
"""

MESH = """
import importlib.util, pathlib, mujoco
root = pathlib.Path(importlib.util.find_spec("myosuite").origin).parent
mesh = root / "simhive" / "myo_sim" / "meshes" / "sacrum.stl"
assert mesh.is_file(), mesh
xml = ("<mujoco><asset><mesh name='m' file='%s'/></asset>"
       "<worldbody><body><geom type='mesh' mesh='m'/></body></worldbody></mujoco>"
       % mesh.as_posix())
m = mujoco.MjModel.from_xml_string(xml)
print("%s: %d vertices, %d faces" % (mesh.name, m.nmeshvert, m.nmeshface))
"""

MYOSUITE_RAW = """
import importlib.util, pathlib, mujoco
root = pathlib.Path(importlib.util.find_spec("myosuite").origin).parent
path = root / "simhive" / "myo_sim" / "body" / "myobody.xml"
m = mujoco.MjModel.from_xml_path(str(path))
print("nq %d, nv %d, nu %d, %d meshes" % (m.nq, m.nv, m.nu, m.nmesh))
"""

SPEC_RAW = """
import importlib.util, pathlib, mujoco
root = pathlib.Path(importlib.util.find_spec("myosuite").origin).parent
path = root / "simhive" / "myo_sim" / "body" / "myobody.xml"
m = mujoco.MjSpec.from_file(str(path)).compile()
print("nq %d, nv %d, nu %d (unedited, through MjSpec)" % (m.nq, m.nv, m.nu))
"""

EDITED = """
import sys
sys.path.insert(0, r"%s")
from myo_curriculum.env import MyoLocomotionEnv
e = MyoLocomotionEnv(stage="W", seed=0)
print(e.describe())
e.close()
""" % V5

STAGES = (
    ("import mujoco", "import mujoco; print(mujoco.__version__)"),
    ("compile a trivial model", TRIVIAL),
    ("compile one MyoSuite mesh", MESH),
    ("compile myobody.xml directly", MYOSUITE_RAW),
    ("compile myobody.xml through MjSpec", SPEC_RAW),
    ("build this project's edited model", EDITED),
)


def clean_environment() -> dict:
    """The same environment with PATH cut back to Windows' own directories.

    Windows resolves a DLL by name against PATH among other places, so another
    program's copy of a library can be loaded into this process in place of the
    one shipped beside the wheel that wants it. On a machine with SCONE
    installed that is a live possibility -- it ships its own physics libraries
    -- and it would look exactly like this: MuJoCo imports, and then its model
    compiler dies with an access violation on a model that cannot fail.
    """
    env = dict(os.environ)
    root = env.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([
        os.path.join(root, "system32"),
        root,
        os.path.join(root, "System32", "Wbem"),
        str(Path(sys.executable).parent),
    ])
    return env


def run(code: str, env=None):
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env)
    lines = [ln for ln in (done.stdout + done.stderr).splitlines() if ln.strip()
             and not ln.startswith("MyoSuite")]
    return done.returncode, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean-path", action="store_true",
                    help="run with PATH cut back to Windows' own directories, "
                         "so no other program's DLLs can be loaded instead")
    args = ap.parse_args(argv)

    print("python  %s" % sys.executable)
    print("version %s" % sys.version.split()[0])
    env = clean_environment() if args.clean_path else None
    print("PATH    %s\n" % ("stripped to Windows' own directories" if env
                            else "as inherited"))

    first_failure = None
    for name, code in STAGES:
        rc, lines = run(code, env)
        if rc == 0:
            print("  ok     %-38s %s" % (name, lines[-1] if lines else ""))
            continue
        # A negative or very large code is a native crash rather than a Python
        # exception; 0xC0000005 is an access violation.
        crash = "" if 0 < rc < 128 else "  (exit %#x -- native crash)" % (rc & 0xFFFFFFFF)
        print("  FAILS  %-38s%s" % (name, crash))
        for line in lines[-3:]:
            print("           %s" % line)
        first_failure = name
        break

    if first_failure is None:
        print("\neverything works on this machine")
        return 0
    print("\nfirst failing stage: %s" % first_failure)
    print("Everything listed above it is fine; that stage is where to look.")
    if first_failure == "import mujoco" and "1114" in " ".join(lines):
        # Easy to hit by building a fresh venv and forgetting the prune, which
        # is per-environment: the plugins live in site-packages.
        print("WinError 1114 at import is the bundled plugins. Run")
        print("    python scripts/prune_mujoco_plugins.py --apply")
        print("against THIS interpreter, then try again.")
    if not args.clean_path:
        print("Try --clean-path next: if the same stage then passes, another "
              "program's DLLs are being loaded from PATH.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
