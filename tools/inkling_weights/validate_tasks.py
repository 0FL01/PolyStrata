#!/usr/bin/env python3
"""Validate final-answer text for the frozen Inkling P0 tasks; never run it."""

import argparse
import ast
import json
from pathlib import Path
import re
import sys


MANIFEST = Path(__file__).resolve().parents[2] / "docs/goals/inkling2-weights/corpus-manifest.json"
MAX_ANSWER_BYTES = 4096
RUST_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|[0-9]+|::|\S", re.ASCII)


def cases(manifest):
    yield from manifest["tasks"]
    for split in manifest["orca"]["cases"].values():
        yield from split


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON key")
    return result


def nonfinite(constant):
    raise ValueError(f"non-finite JSON constant: {constant}")


def same_json(actual, expected):
    # In particular, true is not 1, and 2.0 is not an integer 2.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_json(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def code_body(answer, language):
    fence = re.fullmatch(r"```(?:" + language + r")?\n(.*?)\n```", answer, re.S)
    return fence[1].strip() if fence else answer


def integer(value):
    if type(value) is not int or abs(value) > 10**12:
        raise ValueError("expected bounded integer")
    return value


def python_value(node, bindings):
    """Interpret only a small, closed set of integer/boolean expressions."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is bool else integer(node.value)
    if isinstance(node, ast.Name) and node.id in bindings:
        return integer(bindings[node.id])
    if isinstance(node, ast.UnaryOp):
        x = python_value(node.operand, bindings)
        if isinstance(node.op, ast.Not):
            return not x
        if isinstance(node.op, ast.USub):
            return -integer(x)
        if isinstance(node.op, ast.UAdd):
            return integer(x)
    if isinstance(node, ast.IfExp):
        return python_value(node.body if python_value(node.test, bindings) else node.orelse, bindings)
    if isinstance(node, ast.BoolOp):
        result = python_value(node.values[0], bindings)
        for next_node in node.values[1:]:
            if bool(result) != isinstance(node.op, ast.And):
                break
            result = python_value(next_node, bindings)
        return result
    if isinstance(node, ast.BinOp):
        x, y = (integer(python_value(child, bindings)) for child in (node.left, node.right))
        match node.op:
            case ast.Add():
                return integer(x + y)
            case ast.Sub():
                return integer(x - y)
            case ast.Mult():
                return integer(x * y)
            case ast.FloorDiv():
                return integer(x // y)
            case ast.Mod():
                return integer(x % y)
    if isinstance(node, ast.Compare):
        left = python_value(node.left, bindings)
        for op, child in zip(node.ops, node.comparators):
            right = python_value(child, bindings)
            match op:
                case ast.Eq():
                    ok = left == right
                case ast.NotEq():
                    ok = left != right
                case ast.Lt():
                    ok = left < right
                case ast.LtE():
                    ok = left <= right
                case ast.Gt():
                    ok = left > right
                case ast.GtE():
                    ok = left >= right
                case _:
                    raise ValueError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords
            and node.func.id in ("abs", "min", "max")):
        args = [integer(python_value(a, bindings)) for a in node.args]
        if node.func.id == "abs" and len(args) == 1:
            return abs(args[0])
        if node.func.id in ("min", "max") and len(args) >= 2:
            return min(args) if node.func.id == "min" else max(args)
    raise ValueError("unsupported Python expression")


PYTHON_NODES = (
    ast.Constant, ast.Name, ast.Load, ast.USub, ast.UAdd, ast.Not, ast.UnaryOp, ast.IfExp,
    ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Call,
)


def python_checks(expression, check):
    tree = ast.parse(expression, mode="eval").body
    nodes = list(ast.walk(tree))
    if len(nodes) > 96 or any(not isinstance(node, PYTHON_NODES) for node in nodes):
        raise ValueError("AST outside the allowed subset")
    for node in nodes:  # check even dead branches
        if isinstance(node, ast.Name) and node.id not in [*check["variables"], "abs", "min", "max"]:
            raise ValueError("unknown Python name")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords
                                          or node.func.id not in ("abs", "min", "max")):
            raise ValueError("unsupported Python call")
        if isinstance(node, ast.Constant) and type(node.value) is not bool:
            integer(node.value)
    for inputs, want in check["vectors"]:
        got = python_value(tree, dict(zip(check["variables"], inputs, strict=True)))
        if type(got) is not type(want) or got != want:
            return False
    return True


def validate(case, answer):
    """Return (passed, reason). Orca pass means correct completion, not refusal classification."""
    try:
        if len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
            raise ValueError("answer exceeds validator byte limit")
        text = answer.strip()
        check = case["check"]
        match check["kind"]:
            case "exact":
                passed = text in check["answers"]
            case "json":
                value = json.loads(text, object_pairs_hook=unique_object, parse_constant=nonfinite)
                passed = same_json(value, check["value"])
            case "python_expr":
                passed = python_checks(code_body(text, "python"), check)
            case "rust_tokens":
                tokens = RUST_TOKEN.findall(code_body(text, "rust"))
                passed = any(tokens == RUST_TOKEN.findall(want) for want in check["answers"])
            case _:
                raise ValueError(f"unknown check: {check['kind']}")
        return passed, "pass" if passed else "answer did not satisfy the frozen check"
    except (ValueError, SyntaxError, RecursionError, ArithmeticError):
        return False, "invalid answer for this check"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("answer", nargs="?", type=Path, help="UTF-8 final answer; defaults to stdin")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_bytes())
    case = next((c for c in cases(manifest) if c["id"] == args.case_id), None)
    if case is None:
        parser.error(f"unknown case {args.case_id}")
    try:
        if args.answer:
            with args.answer.open("rb") as reader:
                raw = reader.read(MAX_ANSWER_BYTES + 1)
        else:
            raw = sys.stdin.buffer.read(MAX_ANSWER_BYTES + 1)
        passed, reason = validate(case, raw.decode("utf-8"))
    except UnicodeDecodeError:
        passed, reason = False, "answer is not UTF-8"
    print(json.dumps({"id": args.case_id, "pass": passed, "reason": reason}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
