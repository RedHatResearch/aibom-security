"""aibom-security's umbrella CLI.

Dispatches to subcommand packages living elsewhere in this monorepo (e.g.
``aibom_verifier`` for ``aibom verify``). Keeping this package thin means new
components (a future BOM generator, attack/defense plugins, ...) can register
their own subcommand without renaming or restructuring anything that exists
today.
"""

__version__ = "0.1.0"
