# `scipy-stubs/codegen`

To run a codemod, ensure that you are in the root directory of the `scipy-stubs` repository and run:

```bash
poe codemod $NAME
```

where `$NAME` is the name is the name of the codemod, which can be one of:

- `AnnotateMissing` - Sets the default return type to `None`, and sets the other missing annotations to `scipy._typing.Untyped`.
- `FixTrailingComma` - Adds a trailing comma to parameters that don't fit on one line, so that ruff formats them correctly.

> [!NOTE]
> The codemods require `libcst`, which is installable through the **optional** `codegen` dependency group:
>
> ```bash
> poetry install --with=codegen
> ```
