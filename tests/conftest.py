"""KMP_DUPLICATE_LIB_OK=TRUE for pytest only.

Without it, pytest segfaults during FAISS index init on macOS. numpy
and faiss-cpu each link against their own OpenMP runtime; when both
load in the same process, Intel's OpenMP aborts with "OMP: Error #15:
Initializing libomp.dylib, but found libomp.dylib already initialized."
The env var tells it to continue. Documented caveat: may cause silent
perf or correctness issues for heavy numerical work. For cosine sim on
normalized vectors the practical risk is near zero.

Scoped to tests because bare `import widemem` and `WideMemory()`
outside pytest do not segfault (verified). The collision is specific
to pytest's collection or fixture-ordering path. Setting it library
wide would accept the silent-correctness caveat for every user when
only the test runner needs it.

Revisit if a production user ever reports a segfault on `import
widemem` or first FAISS use. Next step then is
`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` in
widemem/__init__.py, not docs-only.

Breadcrumb: search "OMP: Error #15 libomp already initialized" for
Intel's OpenMP docs and faiss-cpu / pytorch issue threads on the
KMP_DUPLICATE_LIB_OK trade-offs.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
