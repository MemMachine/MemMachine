"""
MemMachine Dify plugin entrypoint.

This module wires up the Dify plugin runtime and starts the plugin server when
executed as a script.
"""

# ruff: noqa: INP001

from dify_plugin import DifyPluginEnv, Plugin

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == "__main__":
    plugin.run()
