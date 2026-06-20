"""Optional graph-memory layer (v1.6 experiment).

Typed entity-relationship triplets extracted alongside flat facts, stored in
SQLite, and traversed at query time to surface relationally-connected memories
that pure vector similarity misses. Targets the open-domain category, where the
flat store is structurally capped.

Fully gated behind MemoryConfig.enable_graph. When off, nothing in this package
is imported on the hot path and behavior is byte-identical to the flat store.
"""
