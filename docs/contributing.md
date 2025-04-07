# Contributing to mmlearn

Thanks for your interest in contributing to mmlearn!

To submit PRs, please fill out the PR template along with the PR. If the PR fixes an issue, please include a link to the
PR to the issue. Below are some details around important things to consider before contributing to the library. A table
of contents also appears below for navigation.

- [Development Practices](#development-practices)
- [Development Requirements](#development-requirements)
- [Coding Guidelines, Formatters, and Checks](#coding-guidelines-formatters-and-checks)
- [Code Documentation](#code-documentation)
- [Tests](#tests)

## Development Practices

We use the standard git development flow of branch and merge to main with PRs on GitHub. At least one member of the core
team needs to approve a PR before it can be merged into main. As mentioned above, tests are run automatically on PRs with
a merge target of main. Furthermore, a suite of static code checkers and formatters are also run on said PRs. These also
need to pass for a PR to be eligible for merging into the main branch of the library. Currently, such checks run on python3.9.

## Development Requirements

For development and testing, we use [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for dependency
management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file.
You may use whatever virtual environment management tool that you would like.
These include uv, conda, and virtualenv.

The easiest way to create and activate a virtual environment for development using uv is:
```bash
uv sync --dev --all-extras
```

Note that the with command is installing all libraries required for the full development workflow. See the `pyproject.toml`
file for additional details as to what is installed with each of these options.

If you need to update the environment libraries, you should change the requirements in the `pyproject.toml` and then update
the `uv.lock` using the command `uv lock`.

## Coding Guidelines, Formatters, and Checks

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code analysis. Ruff checks various rules including
[flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need
to fix before submitting a PR.

Last but not least, we use type hints in our code which are checked using [mypy](https://mypy.readthedocs.io/en/stable/).
The mypy checks are strictly enforced. That is, all mypy checks must pass or the associated PR will not be merge-able.

The settings for `mypy` and `ruff` can be found the `pyproject.toml` files and some standard checks are defined directly
in the `.pre-commit-config.yaml` settings.

All of these checks and formatters are invoked by pre-commit hooks. These hooks are run remotely on GitHub. In order to
ensure that your code conforms to these standards, and, therefore, passes the remote checks, you can install the pre-commit
hooks to be run locally. This is done by running (with your environment active)

```bash
pre-commit install
```

To run the checks, some of which will automatically re-format your code to fit the standards, you can run
```bash
pre-commit run --all-files
```
It can also be run on a subset of files by omitting the `--all-files` option and pointing to specific files or folders.

If you're using VS Code for development, pre-commit should setup git hooks that execute the pre-commit checks each time
you check code into your branch through the integrated source-control as well. This will ensure that each of your commits
conform to the desired format before they are run remotely and without needing to remember to run the checks before pushing
to a remote. If this isn't done automatically, you can find instructions for setting up these hooks manually online.

## Code Documentation

For code documentation, we try to adhere to the [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
For development, __any non-trivial or non-obvious methods added to the library should have a doc string__. For our library
this applies only to code added to the main library in `mmlearn`. Code outside the core library folder, such as tests,
need not incorporate the strict rules of documentation, though clarifying and helpful comments in that code is also
__strongly encouraged__.

__NOTE__: As a matter of convention choice, classes are documented at the "class" level rather than through their `__init__`
functions.

If you are using VS Code a very helpful integration is available to facilitate the creation of properly formatted doc-strings
called autoDocstring [VS Code Page](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) and
[Documentation](https://github.com/NilsJPWerner/autoDocstring). This tool will automatically generate a docstring template
when starting a docstring with triple quotation marks (`"""`). To get the correct format, the following settings should
be prescribed in your VS Code settings JSON:

```json
{
    "autoDocstring.customTemplatePath": "",
    "autoDocstring.docstringFormat": "numpy",
    "autoDocstring.generateDocstringOnEnter": true,
    "autoDocstring.guessTypes": true,
    "autoDocstring.includeExtendedSummary": false,
    "autoDocstring.includeName": false,
    "autoDocstring.logLevel": "Info",
    "autoDocstring.quoteStyle": "\"\"\"",
    "autoDocstring.startOnNewLine": true
}
```

## Tests

All tests for the library are housed in the `tests` folder. The unit and integration tests are run using `pytest`. These
tests are automatically run through GitHub integrations on PRs to the main branch of this repository. PRs that fail any
of the tests will not be eligible to be merged until they are fixed.

To run all tests in the tests folder one only needs to run (with the venv active)
```bash
pytest .
```
To run a specific test with pytest, one runs
```bash
pytest tests/datasets/test_combined_dataset.py
```
