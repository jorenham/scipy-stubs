# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "libcst>=1.5.0",
# ]
# ///

from typing import Final, final
from typing_extensions import override

import libcst as cst
from libcst.codemod import CodemodContext, SkipFile, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor
from libcst.helpers import get_full_name_for_node

_DUNDER_RETURN = {
    "__bool__": "bool",
    "__int__": "int",
    "__float__": "float",
    "__complex__": "complex",
    "__str__": "str",
    "__bytes__": "bytes",
    "__buffer__": "memoryview",
    "__index__": "int",
    "__hash__": "int",
    "__len__": "int",
    "__length_hint__": "int",
    "__repr__": "str",
    "__format__": "str",
    "__init__": "None",
    "__init_subclass__": "None",
    "__set__": "None",
    "__setattr__": "None",
    "__setattribute__": "None",
    "__delattr__": "None",
    "__delattribute__": "None",
    "__delete__": "None",
}


class _BaseMod(VisitorBasedCodemodCommand):
    updated: int

    def __init__(self, /, context: CodemodContext) -> None:
        self.updated = 0
        super().__init__(context)

    @override
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.updated:
            raise SkipFile("unchanged")

        return updated_node


@final
class AnnotateMissing(_BaseMod):
    DESCRIPTION = "Sets the default return type to `None`, and sets the other missing annotations to `scipy._typing.Untyped`."

    untyped: Final[str]

    def __init__(self, /, context: CodemodContext, *, untyped: str = "Untyped") -> None:
        self.untyped = untyped
        super().__init__(context)

    @override
    def leave_Param(self, /, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if updated_node.annotation is not None or updated_node.name.value in {"self", "cls", "_cls"}:
            return updated_node

        AddImportsVisitor.add_needed_import(self.context, "scipy._typing", self.untyped)
        self.updated += 1
        return updated_node.with_changes(annotation=cst.Annotation(cst.Name(self.untyped)))

    @override
    def leave_FunctionDef(self, /, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if updated_node.returns is None:
            self.updated += 1
            return updated_node.with_changes(returns=cst.Annotation(cst.Name("None")))

        if (name := updated_node.name.value) not in _DUNDER_RETURN:
            return updated_node

        return_type = get_full_name_for_node(updated_node.returns.annotation)
        return_type_expect = _DUNDER_RETURN[name]
        if return_type == return_type_expect:
            return updated_node

        if return_type in {self.untyped, "Any"}:
            self.updated += 1
            return updated_node.with_changes(returns=cst.Annotation(cst.Name("None")))

        self.warn(f"{name}() return type is {return_type!r}, not {return_type_expect!r}")

        return updated_node


@final
class FixTrailingComma(_BaseMod):
    DESCRIPTION = "Adds a trailing comma to parameters that don't fit on one line, so that ruff formats them correctly."

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        params = updated_node.params.params

        if (
            # the parameters start on a new line
            isinstance(original_node.whitespace_before_params, cst.ParenthesizedWhitespace)
            # at least 2 parameters
            and len(params) > 1
            # the first parameter has a comma
            and (comma0 := params[0].comma) is not cst.MaybeSentinel.DEFAULT
            # the first comma doesn't end in a newline
            and isinstance(comma0.whitespace_after, cst.SimpleWhitespace)
            # the last parameter has no trailing comma
            and params[-1].comma is cst.MaybeSentinel.DEFAULT
        ):
            self.updated += 1
            return updated_node.with_deep_changes(params[-1], comma=cst.Comma())

        return updated_node
