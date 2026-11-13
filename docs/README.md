# Documentation

This directory holds the source of the MemMachine documentation site,
[docs.memmachine.ai](https://docs.memmachine.ai), built with
[Mintlify](https://www.mintlify.com/).

To preview the site, install the Mintlify CLI as its
[installation guide](https://www.mintlify.com/docs/cli/install) describes,
then run this from the `docs/` directory:

```bash
mint dev
```

Docs CI runs these checks from the same directory:

```bash
mint broken-links
mint openapi-check openapi.json
mint openapi-check platform.openapi.json
```
