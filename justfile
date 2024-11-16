clean:
    rm -rf \
        codegen/*.pyc \
        codegen/__pycache__ \
        scipy-stubs/**/*.pyc \
        scipy-stubs/**/__pycache__ \
        ./**/.mypy_cache \
        ./**/.ruff_cache \
        ./**/.tox

mdformat:
    mdformat \
        CODE_OF_CONDUCT.md \
        CONTRIBUTING.md \
        README.md \
        SECURITY.md \
        tests/README.md

_ruff_format:
    ruff format

_ruff_check:
    ruff check --show-fixes

codespell:
    codespell

repo-review:
    repo-review .

format: mdformat _ruff_format

ruff: _ruff_format _ruff_check

check: codespell repo-review ruff
    mdformat --check

lint: check format

pre-commit:
    pre-commit run --all-files

tox:
    tox -p all

_test_bpr:
    basedpyright tests

_test_mypy:
    mypy --config-file=pyproject.toml tests

typetest: _test_bpr _test_mypy

# Commands with arguments

[positional-arguments]
pyright *paths='scipy-stubs codegen':
    basedpyright {{paths}}

[positional-arguments]
mypy *paths='scipy-stubs codegen':
    mypy --config-file=pyproject.toml {{paths}}

[positional-arguments]
stubtest *modules='scipy':
    stubtest \
        --mypy-config-file=pyproject.toml \
        --allowlist=.mypyignore \
        --ignore-unused-allowlist \
        {{modules}}

[positional-arguments]
codemod name path='scipy-stubs':
    python -m libcst.tool codemod \
        -x \
        --hide-generated-warnings \
        --hide-blacklisted-warnings \
        --hide-progress \
        --include-stubs \
        codegen.mods.{{name}} \
        {{path}}
