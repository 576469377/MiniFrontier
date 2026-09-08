"""Linux seccomp worker for generated pure Python function tests.

Inputs contain candidate code and test arguments, never expected answers.
Load the allowed standard library before a deny-by-default syscall filter.
After loading the filter, files, sockets, subprocesses and other processes
cannot be accessed. This worker must never fall back to unrestricted exec.
"""

import collections
import ctypes
import functools
import itertools
import json
import math
import os
import re
import resource
import sys
from typing import Any


def isolate():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    context = library.seccomp_init(0x80000000)  # SCMP_ACT_KILL_PROCESS
    if not context:
        raise RuntimeError("seccomp_init failed")
    # No filesystem/path lookup, read, network, process creation, ptrace or exec.
    allowed = (
        "write",
        "close",
        "fstat",
        "lseek",
        "brk",
        "mmap",
        "mprotect",
        "munmap",
        "mremap",
        "madvise",
        "futex",
        "clock_gettime",
        "getrandom",
        "getpid",
        "gettid",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "sigaltstack",
        "exit",
        "exit_group",
    )
    for name in allowed:
        number = library.seccomp_syscall_resolve_name(name.encode())
        if number < 0 or library.seccomp_rule_add(context, 0x7FFF0000, number, 0):
            raise RuntimeError("seccomp rule setup failed")
    if library.seccomp_load(context):
        raise RuntimeError("seccomp filter unavailable; code execution is disabled")
    library.seccomp_release(context)


def main():
    payload = json.loads(sys.stdin.buffer.read(128 * 1024))
    os.close(0)
    # Inherited descriptors are closed by the parent, leaving only output pipes.
    try:
        isolate()
    except Exception:
        sys.stdout.write('{"isolation_failed":true}')
        return
    namespace: dict[str, Any] = {
        "__name__": "candidate",
        "math": math,
        "re": re,
        "json": json,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
    }
    result: dict[str, Any]
    try:
        exec(compile(payload["code"], "<candidate>", "exec"), namespace)
        function = namespace[payload["entry_point"]]
        outputs = [
            function(*case.get("args", []), **case.get("kwargs", {})) for case in payload["cases"]
        ]
        result = dict(outputs=outputs)
    except BaseException as error:
        result = dict(error=type(error).__name__)
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
