# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "libcst>=1.5.0",
# ]
# ///

from typing_extensions import override

import libcst as cst
from libcst.codemod import CodemodContext, SkipFile, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor


class AnnotateMissing(VisitorBasedCodemodCommand):
    DESCRIPTION = "Sets the default return type to `None`, and other missing annotations to `scipy._typing.Untyped`"

    updated: int

    def __init__(self, /, context: CodemodContext) -> None:
        self.updated = 0
        super().__init__(context)

    @override
    def leave_Param(self, /, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if updated_node.annotation is not None or updated_node.name.value in {"self", "cls", "_cls"}:
            return updated_node

        AddImportsVisitor.add_needed_import(self.context, "scipy._typing", "Untyped")
        self.updated += 1
        return updated_node.with_changes(annotation=cst.Annotation(cst.Name("Untyped")))

    @override
    def leave_FunctionDef(self, /, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if updated_node.returns is not None:
            return updated_node

        self.updated += 1
        return updated_node.with_changes(returns=cst.Annotation(cst.Name("None")))

    @override
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.updated:
            raise SkipFile("unchanged")

        return updated_node
