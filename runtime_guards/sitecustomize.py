"""CV-only ordinary Python file access guard, loaded by native tool children.

No effect without FINANCE_OPD_PRIVATE_PATHS. This is a leakage regression
guard, not a sandbox against arbitrary native code or subprocess bypasses.
"""
import json
import os
import sys


def _install():
    raw = os.environ.get("FINANCE_OPD_PRIVATE_PATHS")
    if raw is None:
        return
    try:
        paths = json.loads(raw)
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not os.path.isabs(path) or os.path.normpath(path) == os.path.sep
            for path in paths
        ):
            raise ValueError("expected a list of absolute non-root paths")
        roots = tuple({os.path.abspath(path) for path in paths} | {os.path.realpath(path) for path in paths})
    except Exception:
        # Python otherwise reports and ignores sitecustomize import failures.
        os.write(2, b"invalid FINANCE_OPD_PRIVATE_PATHS; refusing unguarded research tools\n")
        os._exit(78)

    def check(path, dir_fd=None):
        if isinstance(path, int):
            try:
                path = os.readlink("/proc/self/fd/" + str(path))
            except OSError:
                return
        elif path is None:
            path = "."
        try:
            path = os.fsdecode(path)
        except TypeError:
            return
        if not os.path.isabs(path) and dir_fd is not None and dir_fd != -1:
            try:
                path = os.path.join(os.readlink("/proc/self/fd/" + str(dir_fd)), path)
            except OSError:
                return
        absolute = os.path.abspath(path)
        for candidate in (absolute, os.path.realpath(absolute)):
            if any(candidate == root or candidate.startswith(root + os.path.sep) for root in roots):
                raise PermissionError("CV-OPD private supervision file access denied")

    def audit(event, args):
        if event in {"open", "os.listdir", "os.scandir"}:
            check(args[0])
        elif event == "os.truncate":
            check(args[0])
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime"}:
            position = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 2,
                        "os.chmod": 2, "os.chown": 3, "os.utime": 3}[event]
            check(args[0], args[position] if len(args) > position else None)
        elif event in {"os.rename", "os.link"}:
            check(args[0], args[2] if len(args) > 2 else None)
            check(args[1], args[3] if len(args) > 3 else None)
        elif event == "os.symlink":
            check(args[1], args[2] if len(args) > 2 else None)

    sys.addaudithook(audit)


_install()
