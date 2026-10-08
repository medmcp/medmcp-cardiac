"""Vendored CineMA model code (MIT) -- the ConvUNetR segmentation network.

Everything here except this file and ``_timm.py`` is written by
``scripts/vendor-cinema.sh`` from the upstream commit recorded in ``UPSTREAM``;
edit upstream, bump the revision, re-run the script. The code is imported only
inside the inference subprocess, never by the MCP server.

Upstream: https://github.com/mathpluscode/CineMA (Fu et al., Communications
Medicine 2026). License: MIT, see ``LICENSE`` in this directory.
"""
