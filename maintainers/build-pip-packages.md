# Building Pip Packages for MemMachine

This document covers building the Python packages, for PyPI or for local
installation.

## Packages

Each package has its own directory under `packages/`:

- `packages/common`: `memmachine-common`, the API models the client and server
  share.
- `packages/client`: `memmachine-client`, the REST client.
- `packages/server`: `memmachine-server`, the server.
- `packages/meta`: `memmachine`, a meta-package that depends on the client and
  the server.

## Building

From the repository root, build each package with `uv build`. The distribution
files go to `dist/` in the repository root.

```bash
uv build --project packages/common
uv build --project packages/client
uv build --project packages/server
uv build --project packages/meta
```

To install a locally built package, install its wheel together with the
`memmachine-common` wheel it depends on, for example:

```bash
pip install dist/memmachine_common-*.whl dist/memmachine_client-*.whl
```

## Versions and Publishing

Each package's version comes from the repository's git tags, through
setuptools-scm. Publishing a GitHub release runs
[pypi-publish.yml](../.github/workflows/pypi-publish.yml), which builds the
packages and uploads them to PyPI.
