from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EgoDependencyTests(unittest.TestCase):
    def test_ego_import_and_help_do_not_require_pi3(self) -> None:
        """The added DA3 entry point must import without the ignored checkout."""
        probe = r'''
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("track4world.nets.external.pi3"):
        raise ModuleNotFoundError("Pi3 imports are blocked for this probe")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

import demo_3dff_ego as demo
from track4world.nets.model_3dff_ego import Track4World3DFFEgo

parser = demo.build_parser()
coordinate_action = next(
    action for action in parser._actions if action.dest == "coordinate"
)
assert tuple(coordinate_action.choices) == ("world_depthanythingv3",)
args = parser.parse_args(["--H", "16", "--S", "8"])
assert args.coordinate == "world_depthanythingv3"

try:
    Track4World3DFFEgo(use_model="pi3")
except ValueError as exc:
    assert "depthanythingv3" in str(exc)
else:
    raise AssertionError("A non-DA3 Ego-centric model request was accepted")
'''
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(ROOT)
            if not existing_pythonpath
            else os.pathsep.join((str(ROOT), existing_pythonpath))
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Pi3-free import probe failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
