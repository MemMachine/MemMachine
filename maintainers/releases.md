# Releases

Publishing a GitHub release runs the workflows that upload the Python packages
to PyPI, the TypeScript client and the OpenClaw plugin to npm, and the images to
Docker Hub. GitHub's
[Managing releases in a repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
covers creating and deleting a release.

- **Tag:** a three-part version, `MAJOR.MINOR.PATCH`, with an optional `v`
  prefix and an optional pre-release suffix, such as `v0.4.0` or `v0.5.0-rc1`.
  The npm publish workflows reject other forms.
- **Version number:** while the major version is 0, changing the minor version,
  such as `0.4.1` to `0.5.0`, marks a breaking release. Breaking changes go only
  into such a release; a patch release stays compatible.
- **Pre-releases:** package versions come from the tag, so PyPI and npm get a
  pre-release only when the tag has a pre-release suffix. GitHub's "Set as a
  pre-release" box only labels the release on GitHub.
- **Mistakes:** deleting a release and its tag doesn't remove what its workflows
  uploaded, and PyPI and npm won't accept the same version again, so a corrected
  release needs a new version number.
