<!-- omit in toc -->

# Contributing to scipy-stubs

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued.
See the [Table of Contents](#table-of-contents) for different ways to help and
details about how this project handles them.
Please make sure to read the relevant section before making your contribution.
It will make it a lot easier for us maintainers and smooth out the experience
for all involved.
The community looks forward to your contributions. 🎉

> [!NOTE]
> And if you like scipy-stubs, but just don't have time to contribute, that's fine.
> There are other easy ways to support the project and show your appreciation,
> which we would also be very happy about:
>
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)

## Code of Conduct

This project and everyone participating in it is governed by the
[`scipy-stubs` Code of Conduct][coc].
By participating, you are expected to uphold this code.
Please report unacceptable behavior to `jhammudoglu<at>gmail<dot>com`.

## I Have a Question

> [!NOTE]
> If you want to ask a question, we assume that you have read the
> available [Documentation][doc].

Before you ask a question, it is best to search for existing [Issues][bug]
that might help you.
In case you have found a suitable issue and still need clarification,
you can write your question in this issue.
It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Open an [Issue][bug].
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (Python, mypy, pyright, ruff, etc),
  depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
>
> When contributing to this project,
> you must agree that you have authored 100% of the content,
> that you have the necessary rights to the content and that the content you
> contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information.
Therefore, we ask you to investigate carefully, collect information and
describe the issue in detail in your report.
Please complete the following steps in advance to help us fix any potential
bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g.
  using incompatible environment components/versions
  (Make sure that you have read the [documentation][doc].
  If you are looking for support, you might want to check
  [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved)
  the same issue you are having, check if there is not already a bug report
  existing for your bug or error in the [bug tracker][bug].
- Also make sure to search the internet (including Stack Overflow) to see if
  users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment,
    package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue?
    And can you also reproduce it with older versions?

<!-- omit in toc -->

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs
> including sensitive information to the issue tracker, or elsewhere in public.
> Instead sensitive bugs must be sent by email to `jhammudoglu<at>gmail<dot>com`.

We use GitHub issues to track bugs and errors.
If you run into an issue with the project:

- Open an [Issue][bug].
  (Since we can't be sure at this point whether it is a bug or not,
  we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the
  *reproduction steps* that someone else can follow to recreate the issue on
  their own.
  This usually includes your code.
  For good bug reports you should isolate the problem and create a reduced test
  case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps.
  If there are no reproduction steps or no obvious way to reproduce the issue,
  the team will ask you for those steps and mark the issue as `needs-repro`.
  Bugs with the `needs-repro` tag will not be addressed until they are
  reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`,
  as well as possibly other tags (such as `critical`), and the issue will be
  left to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for
scipy-stubs, **including completely new features and minor improvements to existing
functionality**.
Following these guidelines will help maintainers and the community to
understand your suggestion and find related suggestions.

<!-- omit in toc -->

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation][doc] carefully and find out if the functionality is
  already covered, maybe by an individual configuration.
- Perform a [search][bug] to see if the enhancement has already been suggested.
  If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project.
  It's up to you to make a strong case to convince the project's developers of
  the merits of this feature. Keep in mind that we want features that will be
  useful to the majority of our users and not just a small subset. If you're
  just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues][bug].

- Use a **clear and descriptive title** for the issue to identify the
  suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as
  many details as possible.
- **Describe the current behavior** and **explain which behavior you expected
  to see instead** and why. At this point you can also tell which alternatives
  do not work for you.
- **Explain why this enhancement would be useful** to most scipy-stubs users.
  You may also want to point out the other projects that solved it better and
  which could serve as inspiration.

### Your First Code Contribution

Ensure you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed.
Now you can install the project with the dev dependencies:

```bash
uv sync --frozen  --python=3.13
```

> [!NOTE]
> Python 3.13 is required to properly run `mdformat` (see the documentation [note here][mdformat-3.13]).

### pre-commit

`scipy-stubs` uses [pre-commit](https://pre-commit.com/) to ensure that the code is
formatted and typed correctly when committing the changes.

```bash
poe pre-commit
```

> [!NOTE]
> Pre-commit doesn't run `stubtest`. This will be run by github actions
> when submitting a pull request. See the next section for more details.

### Tox

The pre-commit hooks and `stubtest` can easily be run with [tox](https://github.com/tox-dev/tox).
It can be installed with:

```bash
uv tool install tox --with tox-uv
```

To run all environments in parallel, run:

```shell
$ tox -p all
repo-review: OK ✔ in 0.4 seconds
3.12: OK ✔ in 10.38 seconds
3.10: OK ✔ in 10.62 seconds
3.11: OK ✔ in 11.04 seconds
3.13: OK ✔ in 19.42 seconds
  repo-review: OK (0.40=setup[0.04]+cmd[0.36] seconds)
  pre-commit: OK (24.91=setup[0.04]+cmd[24.87] seconds)
  3.10: OK (10.62=setup[0.11]+cmd[10.51] seconds)
  3.11: OK (11.04=setup[0.04]+cmd[11.00] seconds)
  3.12: OK (10.38=setup[0.04]+cmd[10.34] seconds)
  3.13: OK (19.42=setup[0.04]+cmd[19.38] seconds)
  congratulations :) (24.96 seconds)
```

### Improving The Documentation

All [documentation] lives in the `README.md`. Please read it carefully before
proposing any changes. Ensure that the markdown is formatted correctly with
[markdownlint](https://github.com/DavidAnson/markdownlint/tree/main).

## Useful resources

- [official typing docs](https://typing.readthedocs.io/en/latest/)
- [`basedmypy` docs](https://kotlinisland.github.io/basedmypy/)
- [`basedpyright` docs](https://docs.basedpyright.com/latest/)
- [`basedpyright` playground](https://basedpyright.com/)
- [`numpy.typing` docs](https://numpy.org/doc/stable/reference/typing.html)
- [`scipy-stubs` docs](https://github.com/jorenham/scipy-stubs/blob/master/README.md)
- [`typing_extensions` docs](https://typing-extensions.readthedocs.io/en/latest/#)

<!-- omit in toc -->

## Attribution

This guide is based on the **contributing-gen**.
[Make your own](https://github.com/bttger/contributing-gen)!

[bug]: https://github.com/jorenham/scipy-stubs/issues
[coc]: https://github.com/jorenham/scipy-stubs/blob/master/CODE_OF_CONDUCT.md
[doc]: https://github.com/jorenham/scipy-stubs?tab=readme-ov-file#scipy-stubs
[mdformat-3.13]: https://mdformat.readthedocs.io/en/stable/users/configuration_file.html#exclude-patterns
