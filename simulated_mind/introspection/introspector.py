"""Introspector: read Python code via AST and return structured snapshots.

This MVP implementation focuses on walking an AST to list all module-level
classes and functions with their line spans. More sophisticated analyses can be
added later (docstring extraction, decorators, etc.).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List

from ..logging.journal import Journal


@dataclass(frozen=True)
class CodeSymbol:
    """A discovered symbol inside a Python module."""

    qualname: str  # e.g. "mypkg.mymodule.MyClass.method"
    node_type: str  # "class" | "function"
    lineno: int
    end_lineno: int | None


class _SymbolVisitor(ast.NodeVisitor):
    """AST visitor that records fully-qualified names of classes & functions."""

    def __init__(self, module_qualname: str):
        self._module_qualname = module_qualname
        self.symbols: list[CodeSymbol] = []
        self._scope: list[str] = []

    # Helpers -----------------------------------------------------------------

    def _qualname_for(self, name: str) -> str:
        scope = ".".join(self._scope)
        return ".".join([p for p in [self._module_qualname, scope, name] if p])

    # Visitor methods ---------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef):
        qualname = self._qualname_for(node.name)
        self.symbols.append(
            CodeSymbol(
                qualname=qualname,
                node_type="class",
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
            )
        )
        # Walk into class scope
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802, N802
        qualname = self._qualname_for(node.name)
        self.symbols.append(
            CodeSymbol(
                qualname=qualname,
                node_type="function",
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
            )
        )
        # Visit nested functions
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


class Introspector:
    """Utility class for code introspection using Python AST with transparency logging."""

    def __init__(self, journal: Journal | None = None):
        self.journal = journal or Journal.null()

    def snapshot(self, module_path: Path) -> List[CodeSymbol]:
        if not module_path.exists():
            raise FileNotFoundError(module_path)
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        module_qualname = self._module_qualname(module_path)
        visitor = _SymbolVisitor(module_qualname)
        visitor.visit(tree)
        symbols = visitor.symbols
        self.journal.log_event(
            "introspector.snapshot",
            {"module": str(module_path), "symbol_count": len(symbols)},
        )
        return symbols

    # ----------------------------------------------------------------------
    # Insertion-Point Discovery (MVP)
    # ----------------------------------------------------------------------

    def find_insertion_points(self, module_path: Path) -> List[CodeSymbol]:
        """Return symbols whose body only contains a single `pass` (stub)."""
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
        module_qualname = self._module_qualname(module_path)
        candidates: list[CodeSymbol] = []

        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if body and len(body) == 1 and isinstance(body[0], ast.Pass):
                name = getattr(node, "name", None)
                if name:
                    qualname = ".".join([module_qualname, name])
                    candidates.append(
                        CodeSymbol(
                            qualname=qualname,
                            node_type=node.__class__.__name__.lower(),
                            lineno=node.lineno,
                            end_lineno=getattr(node, "end_lineno", None),
                        )
                    )
        self.journal.log_event(
            "introspector.insertion_points",
            {"module": str(module_path), "candidates": len(candidates)},
        )
        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _module_qualname(path: Path) -> str:
        """Convert a file path to a dotted module path (best-effort)."""
        parts: list[str] = []
        for part in path.with_suffix("").parts[::-1]:
            if part == "":
                break
            parts.insert(0, part)
            if part == "simulated_mind":
                break  # stop at project root package
        return ".".join(parts)
