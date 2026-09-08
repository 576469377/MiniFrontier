"""Bounded, resettable local tool tasks. Files and state are entirely in memory."""

import copy
import json
from pathlib import PurePosixPath

from minifrontier.training.verifiers import run_python

VERSION = "local-tools-v1"
TOOLS = {"lookup", "search", "read_file", "replace_text", "set_value", "python"}
ARGUMENTS: dict[str, dict[str, str | None]] = {
    "lookup": dict(table="string", where="object", columns="array"),
    "search": dict(query="string"),
    "read_file": dict(path="string"),
    "replace_text": dict(path="string", old="string", new="string"),
    "set_value": dict(key="string", value=None),
    "python": dict(code="string", entry_point="string", cases="array"),
}


def definitions(enabled):
    descriptions = {
        "lookup": "Query a local table using exact column equality; select columns, at most 32 rows.",
        "search": "Search local documents by query terms; return at most three matching documents.",
        "read_file": "Read a named file from this task's simulated project.",
        "replace_text": "Replace exactly one occurrence in an allowed simulated project file.",
        "set_value": "Set an allowed key in this task's simulated state.",
        "python": "Run a pure Python function on cases with args/kwargs; return its JSON results.",
    }
    return [
        dict(
            name=name,
            description=descriptions[name],
            parameters=dict(
                type="object",
                properties={
                    key: dict(type=kind) if kind else {} for key, kind in ARGUMENTS[name].items()
                },
                required=list(ARGUMENTS[name]),
                additionalProperties=False,
            ),
        )
        for name in sorted(enabled)
    ]


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def safe_path(path):
    return (
        isinstance(path, str)
        and bool(path)
        and not PurePosixPath(path).is_absolute()
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


class ToolEnvironment:
    def __init__(self, specification):
        if len(json_bytes(specification)) > 256 * 1024:
            raise ValueError("tool environment exceeds the 256 KiB initial-state budget")
        self.specification = copy.deepcopy(specification)
        if not isinstance(specification.get("tools"), list) or any(
            not isinstance(name, str) for name in specification["tools"]
        ):
            raise ValueError("tool names must be a list of strings")
        self.enabled = set(specification.get("tools", []))
        if not self.enabled or not self.enabled <= TOOLS:
            raise ValueError("unknown or empty local tool set")
        self.max_calls = specification.get("max_calls", 8)
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 8:
            raise ValueError("tool call budget must be an integer in [1,8]")
        self.files = copy.deepcopy(specification.get("files", {}))
        self.state = copy.deepcopy(specification.get("state", {}))
        if not isinstance(self.state, dict) or not isinstance(self.files, dict):
            raise ValueError("tool state/files must be objects")
        if any(not safe_path(p) or not isinstance(v, str) for p, v in self.files.items()):
            raise ValueError("simulated files need relative paths and text content")
        for field in ("writable_keys", "writable_files"):
            values = specification.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError("writable names must be a list of strings")
        if any(path not in self.files for path in specification.get("writable_files", [])):
            raise ValueError("writable simulated files must exist in the initial state")
        tables, documents = specification.get("tables", {}), specification.get("documents", [])
        if not isinstance(tables, dict) or any(
            not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            for rows in tables.values()
        ):
            raise ValueError("tables must map names to rows of objects")
        if not isinstance(documents, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("text"), str)
            for row in documents
        ):
            raise ValueError("documents require string id/text fields")
        self.calls = self.errors = 0

    def execute(self, text):
        """Strict structure before dispatch; a malformed batch cannot partially execute."""
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict) or set(payload) - {"calls", "content"}:
                raise ValueError("expected calls/content object")
            calls = payload["calls"]
            if not isinstance(calls, list) or not 1 <= len(calls) <= 4:
                raise ValueError("one to four calls per turn required")
            if self.calls + len(calls) > self.max_calls:
                raise ValueError("tool call budget exhausted")
            normalized = []
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("each call must be an object")
                if "function" in call:
                    if set(call) - {"id", "type", "function"} or call.get("type") != "function":
                        raise ValueError("invalid function envelope")
                    call = call["function"]
                if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                    raise ValueError("each call needs exactly name and arguments")
                name, arguments = call["name"], call["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(name, str) or name not in self.enabled:
                    raise ValueError("tool is not enabled by this task")
                if not isinstance(arguments, dict) or len(json_bytes(arguments)) > 64 * 1024:
                    raise ValueError("tool arguments must be a bounded object")
                normalized.append((name, arguments))
        except (ValueError, KeyError, TypeError) as error:
            self.errors += 1
            return [dict(error=str(error), kind="invalid_call")]
        results = []
        for name, arguments in normalized:
            self.calls += 1
            # Restore simulated mutations if an operation or output limit fails.
            before = copy.deepcopy((self.state, self.files))
            try:
                value = self.dispatch(name, arguments)
                if len(json_bytes(value)) > 8192:
                    raise ValueError("tool output exceeds 8192 bytes")
                if len(json_bytes([self.state, self.files])) > 256 * 1024:
                    raise ValueError("tool state exceeds 256 KiB")
                results.append(dict(tool=name, output=value))
            except (ValueError, KeyError, TypeError, IndexError) as error:
                self.state, self.files = before
                self.errors += 1
                results.append(dict(tool=name, error=str(error)))
        return results

    def dispatch(self, name, args):
        required = set(ARGUMENTS[name])
        if set(args) != required:
            raise ValueError("arguments do not match the tool schema")
        if name == "lookup":
            where, columns = args["where"], args["columns"]
            if not isinstance(where, dict) or not isinstance(columns, list) or not columns:
                raise ValueError("lookup needs an exact-match object and nonempty column list")
            if any(not isinstance(c, str) for c in columns):
                raise ValueError("column names must be strings")
            rows = self.specification.get("tables", {})[args["table"]]
            return [
                {column: row[column] for column in columns}
                for row in rows
                if all(row.get(key) == value for key, value in where.items())
            ][:32]
        if name == "search":
            if not isinstance(args["query"], str) or not args["query"].strip():
                raise ValueError("search requires nonempty text")
            terms = args["query"].casefold().split()
            documents = self.specification.get("documents", [])
            scored = [
                (sum(term in row["text"].casefold() for term in terms), index, row)
                for index, row in enumerate(documents)
            ]
            return [row for score, _, row in sorted(scored, key=lambda v: (-v[0], v[1])) if score][
                :3
            ]
        if name in {"read_file", "replace_text"}:
            path = args["path"]
            if not safe_path(path):
                raise ValueError("invalid simulated path")
            if name == "read_file":
                return self.files[path]
            if path not in self.specification.get("writable_files", []):
                raise ValueError("file is not writable in this task")
            old, new = args["old"], args["new"]
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError("replacement requires nonempty old text and string new text")
            if self.files[path].count(old) != 1:
                raise ValueError("replacement must match exactly one occurrence")
            self.files[path] = self.files[path].replace(old, new, 1)
            return dict(changed=True)
        if name == "set_value":
            if args["key"] not in self.specification.get("writable_keys", []):
                raise ValueError("state key is not writable in this task")
            self.state[args["key"]] = args["value"]
            return dict(changed=True)
        if not isinstance(args["code"], str) or not isinstance(args["entry_point"], str):
            raise ValueError("Python code and entry point must be strings")
        if (
            not isinstance(args["cases"], list)
            or not args["cases"]
            or any(not isinstance(c, dict) or set(c) - {"args", "kwargs"} for c in args["cases"])
        ):
            raise ValueError("Python tool accepts bounded arguments, never test expectations")
        result = run_python(args["code"], args["entry_point"], args["cases"])
        if result["reason"] != "completed":
            raise ValueError("isolated Python failed: " + result["reason"])
        return result["outputs"]
