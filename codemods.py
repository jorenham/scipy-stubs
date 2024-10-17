# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "libcst>=1.5.0",
# ]
# ///

from typing_extensions import override

import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor


class AnnotateMissing(VisitorBasedCodemodCommand):
    DESCRIPTION: str = "Sets the default return type to `None`, and other missing annotations to `scipy._typing.Untyped`"

    @override
    def leave_Param(self, /, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if updated_node.annotation is not None or updated_node.name.value in {"self", "cls"}:
            return updated_node

        AddImportsVisitor.add_needed_import(self.context, "scipy._typing", "Untyped")
        return updated_node.with_changes(annotation=cst.Annotation(cst.Name("Untyped")))

    @override
    def leave_FunctionDef(self, /, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if updated_node.returns is not None:
            return updated_node

        return updated_node.with_changes(returns=cst.Annotation(cst.Name("None")))
