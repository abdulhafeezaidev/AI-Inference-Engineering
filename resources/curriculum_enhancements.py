# Curriculum enhancements: modern AI-assisted engineering, unfamiliar codebase comprehension,
# diagnostic debugging, security hardening, and 2026+ interview formats (Google, Meta, Anthropic, Stripe).

PREDICT_AND_TRACE_DAYS = {
    2, 3, 4, 5, 8, 15, 16, 17, 18, 22, 25, 30, 34, 36, 39, 40, 41, 50, 57, 60,
    65, 72, 79, 114, 117, 124, 141, 142, 145, 149, 150, 152
}

PREDICT_AND_TRACE_TRAPS = {
    2: "shallow-copy aliasing where modifying nested mutable config silently corrupts original reference",
    3: "positional-only parameter passed as keyword raising TypeError at runtime",
    4: "unsound TypeGuard returning True on dict with missing key or wrong value type",
    5: "structural match/case fall-through with unhandled enum variant reaching assert_never",
    8: "mutable default argument in dataclass without default_factory sharing state across instances",
    15: "return inside finally block silently swallowing an active exception and breaking error propagation",
    16: "generator @contextmanager missing try/finally around yield leaking open file descriptors",
    17: "ExceptionGroup caught with bare except instead of except*, losing sub-exception tracebacks",
    18: "raise exc from None silently obliterating root cause exception diagnostic context",
    22: "re-iterating over an exhausted generator silently returning an empty sequence without error",
    25: "itertools.tee unconsumed sibling iterator causing unbounded internal memory buffering",
    30: "mutating an object's hashed field after inserting into a dict causing key lookup failure",
    34: "strong reference inside closure callback attached to weakref.finalize preventing garbage collection",
    36: "late-binding closures inside a loop: [lambda: i for i in range(5)] evaluating to 4 for all items",
    39: "descriptor storing state on descriptor self instead of instance.__dict__, sharing state across instances",
    40: "__set_name__ attribute name mismatch causing silent attribute shadowing and property lookup failure",
    41: "@cached_property writing to instance.__dict__ on a __slots__ class that omits '__dict__'",
    50: "binary struct.pack endianness mismatch (> big-endian vs < little-endian) corrupting float32 bytes",
    57: "creating unreferenced asyncio.create_task that gets garbage-collected mid-execution",
    60: "catching asyncio.CancelledError as generic Exception and failing to re-raise, breaking cancellation",
    65: "async generator terminating without awaiting cleanup in aclose() leaking event loop resources",
    72: "partial TCP read assuming reader.read(n) returns exactly n bytes instead of <= n bytes",
    79: "submitting blocking I/O to default thread pool causing thread pool starvation for other tasks",
    114: "metaclass __new__ omitting **kwargs forwarded to super().__new__ breaking cooperative inheritance",
    117: "__init_subclass__ omitting super().__init_subclass__(**kwargs) dropping subclass arguments",
    124: "AST transformer modifying AST nodes in-place without calling ast.fix_missing_locations()",
    141: "sys.getrefcount returning unexpected count due to argument passing into inspection frame",
    142: "circular reference with custom __del__ preventing garbage collector cyclic reclamation",
    145: "writing to a memoryview slicing an immutable bytes object raising BufferError",
    149: "in-place broadcast view mutation modifying multiple logical elements simultaneously",
    150: "stride=0 broadcast views behaving unexpectedly under contiguous-memory assumptions",
    152: "integer overflow or signed/unsigned underflow in INT8 quantization clamping calculation",
}

MICRO_BUG_DRILLS = {
    2: "shallow-copy aliasing mutation corrupting shared nested inference config",
    4: "unsound TypeGuard narrowing on missing dictionary keys causing unexpected KeyError",
    6: "argparse missing required positional argument handling returning exit code 1 instead of 2",
    8: "mutable default in dataclass without default_factory sharing state across instances",
    10: "structural pattern matching fall-through with unhandled enum variant",
    12: "invariant generic type variance error rejecting valid protocol subtype",
    14: "mutable reference leak across simulated worker boundary",
    16: "generator @contextmanager missing try/finally around yield leaking file descriptors",
    18: "raise exc from None masking root cause in chained exception translation",
    20: "test double fake clock failing to advance time under nested timer callbacks",
    22: "generator function resuming after exhaustion raising unexpected RuntimeError",
    24: "itertools.tee memory leak from unconsumed sibling iterator",
    26: "ExitStack early return bypassing transactional rollback on partial write failure",
    28: "PR diff leaky abstraction and 5+ line duplicated code churn",
    30: "mutating an object after inserting into a dict causing key lookup failure",
    32: "custom __hash__ implementation inconsistent with __eq__ on float attributes",
    34: "weakref strong reference inside closure callback preventing GC reclamation",
    36: "late-binding loop variable capture inside closure list",
    38: "decorator omitting functools.wraps stripping metadata and type annotations",
    40: "descriptor storing state on descriptor instance instead of instance dict",
    42: "descriptor attribute shadowing and lookup order collision",
    44: "heapq tuple comparison crash when priorities are equal and items lack < operator",
    46: "LRU cache doubly-linked list node pointer detachment bug on eviction",
    48: "graph cycle detection failing on disconnected graph components",
    50: "binary struct unpack endianness mismatch decoding corrupt IEEE-754 floats",
    52: "SafeTensors 64-bit JSON header length parsed with system endianness instead of little-endian uint64",
    54: "mmap zero-copy slice boundary off-by-one overflowing mapped buffer",
    56: "4-level progressive container state corruption under concurrent read/write",
    58: "garbage-collected unreferenced asyncio.create_task dropping background job",
    60: "catching asyncio.CancelledError without re-raising, preventing clean cancellation",
    62: "asyncio.to_thread passing mutable shared objects without thread synchronization locks",
    64: "async iterator __anext__ hanging due to unhandled exception inside task",
    66: "async queue deadlock due to missing task_done() call",
    68: "InvalidStateError when setting result on a timed-out caller Future",
    70: "dynamic batcher race condition under bursty concurrent arrivals",
    72: "partial TCP read assuming reader.read(n) returns n bytes instead of chunking",
    74: "raw ASGI server sending response headers twice or omitting http.response.body",
    76: "dropping active SSE streams abruptly without terminal events",
    78: "ThreadPoolExecutor unhandled worker exception swallowed without future.result()",
    80: "putting unpicklable generator or lambda into multiprocessing.Queue",
    82: "omitting shm.close() and shm.unlink() leaking shared memory segments across worker crashes",
    84: "git bisect isolating latency regression in multi-process streaming service",
    86: "dynamic strategy registry silently overwriting existing handler on duplicate key",
    88: "randomized property-based test failure on zero-length edge case input",
    90: "cyclic imports and missing __all__ export curation in package entrypoint",
    92: "SQLite concurrent write omitting busy_timeout causing database is locked error",
    94: "non-constant-time token comparison vulnerable to timing side-channel attacks",
    96: "eager string formatting in logging call degrading benchmark throughput",
    98: "SQLite WAL checkpoint contention under simulated high write concurrency",
    100: "circular dependency injection cycle in lightweight DI container causing recursion crash",
    102: "stateful property machine transition violating balance invariant",
    104: "non-deterministic chaos test failing due to uncoordinated sleep timers",
    106: "capstone worker crash leaving orphaned child processes without SIGTERM handling",
    108: "priority inversion in request queue under mixed streaming workloads",
    110: "slow consumer backpressure causing unbounded async memory queue growth",
    112: "capstone clean-room verification and subtle worker IPC protocol desynchronization",
    114: "metaclass type.__new__ omitting **kwargs forwarded to superclass",
    116: "class decorator stripping __annotations__ and breaking runtime type introspection",
    118: "MiniModule parameter registration failing to recurse into nested submodules",
    120: "signature introspection failing on wrapped functions without __wrapped__ unwrapping",
    122: "AST transformer failing to call ast.fix_missing_locations() causing compilation crash",
    124: "bytecode analyzer failing to handle EXTENDED_ARG opcodes in Python 3.12+",
    126: "@jit_trace graph extractor failing on dynamic control flow branches",
    128: "code-generated validator variable name shadowing breaking nested validation",
    130: "custom serializer in schema compiler failing to handle nested None values",
    132: "non-atomic pointer updates during configuration reload under concurrent threads",
    134: "FastAPI route registry regex collision matching incorrect endpoint handler",
    136: "SSRF vulnerability via unvalidated redirect URL in HTTP request handler",
    138: "async dependency injection yield before cleanup causing orphaned database session",
    140: "metaprogramming framework route dispatch regression under load",
    142: "cyclic reference with custom __del__ preventing garbage collector cycle reclamation",
    144: "modifying a buffer exported by an immutable bytes object raising BufferError",
    146: "zero-copy ring buffer slicing into bytes triggering unexpected heap allocations",
    148: "row-major (C-order) vs column-major (Fortran-order) stride offset indexing mismatch",
    150: "zero-stride broadcast view failing when passed to contiguous-memory operations",
    152: "integer overflow in symmetric INT8 quantization clamping calculation",
    154: "autograd tape backward pass invalidated by in-place tensor mutation",
    156: "KV-cache buffer index off-by-one overwriting last prompt token key during decode",
    158: "Rotary Positional Embeddings applied to values instead of queries and keys",
    160: "BPE tokenizer corrupting multi-byte UTF-8 sequences at token boundaries",
    162: "FlashAttention online softmax normalization scale applied after exponentiation",
    164: "PagedAttention physical block table logical-to-physical mapping failing on block boundary",
    166: "continuous batching scheduler preemption evicting highest-priority request instead of lowest",
    168: "PagedAttention block allocator memory leak under simulated preemption",
    170: "cProfile profiling overhead skewing recursive function self-time calculations",
    172: "Py-Spy sampling profiler missing symbols preventing C-frame unwinding",
    174: "unsafe micro-optimization violating IEEE-754 floating point associativity",
    176: "memory leak from cyclic reference involving circular closure callbacks",
    178: "race condition on unsynchronized shared counter in free-threaded Python (PEP 703)",
    180: "C99 fast_ops.c buffer-alignment and pointer-lifetime boundary defect",
    182: "native C kernel ctypes binding failing under multi-threaded GIL release",
    184: "C++ paged_attention_binding.cpp memoryview lifetime premature GC collection defect",
    186: "Triton GPU kernel block stride arithmetic out-of-bounds on non-power-of-two shapes",
    188: "CUDA event recorded on incorrect stream causing synchronization race",
    190: "UNIX domain socket recv_into buffer truncation on IPC messages exceeding 4096 bytes",
    192: "Tensor Parallelism all-reduce sum reduction missing rank normalization",
    194: "distributed rendezvous barrier deadlock when one worker rank encounters an error",
    196: "distributed IPC socket buffer deadlock under heavy message burst",
    198: "Prometheus histogram exposition failing due to non-monotonically increasing bucket bounds",
    200: "circuit breaker half-open state allowing unlimited concurrent requests instead of single probe",
    202: "hardcoded port numbers in distributed chaos tests causing socket collisions",
    204: "vLLM async LLM engine request queue drop under burst arrival",
    206: "OpenAI /v1/chat/completions protocol chunk framing missing terminal [DONE] SSE marker",
    208: "LoRA multi-adapter LRU cache manager evicting active adapter while in-flight decode step runs",
    210: "vllm_mini continuous scheduler hanging on chunked prefill boundary",
    212: "prefix-aware load balancer tie-breaker routing all requests to a single worker instance",
    214: "speculative decoding draft token verification accepting rejected token continuation",
    216: "masking worker process OOM crash exit codes as clean shutdowns",
    218: "capstone multi-worker engine deadlocking when worker process crashes mid-batch",
    220: "benchmark client including initial warmup queries in Time-To-First-Token percentiles",
    222: "SIGINT signal handler failing to gracefully drain active async SSE streams",
    224: "capstone end-to-end continuous batching verification under maximum context pressure",
}

MODE_B_AI_AUDITS = {
    34: (
        "Audit an AI-generated weakref cache implementation for cyclic reference leaks; catch the AI leaving strong references to the target instance inside closure callbacks attached to weakref.finalize.",
        "Write a tracemalloc script proving the finalized target object is retained in memory due to the closure capture, and refactor using a detached static cleanup function."
    ),
    41: (
        "Audit an AI-generated lazy attribute caching pattern; catch the AI attempting to write directly to instance.__dict__ on a __slots__ class that does not declare '__dict__'.",
        "Write a failing pytest proving AttributeError: 'SlottedClass' object has no attribute '__dict__' and implement descriptor-level descriptor storage or declare __dict__ explicitly."
    ),
    48: (
        "Audit an AI-generated topological sort and DAG cycle detector; catch the AI failing on disconnected components or confusing cross-edges with back-edges in directed graphs.",
        "Write a failing pytest proving false cycle detection on a diamond DAG with disconnected roots, then implement proper 3-color (white/gray/black) DFS."
    ),
    55: (
        "Audit an AI-generated structured logging middleware with ContextVar correlation IDs; catch the AI leaking request context across async task boundaries or logging unredacted authorization tokens.",
        "Write a failing pytest demonstrating cross-request context contamination, and enforce contextcopy / reset tokens in async task lifecycles."
    ),
    62: (
        "Audit an AI-generated blocking call wrapper; catch the AI passing a mutable shared list to asyncio.to_thread without acquiring a threading.Lock, causing race condition corruption.",
        "Write a multi-threaded stress test demonstrating lost updates under high concurrency, and wrap shared access in threading.Lock."
    ),
    69: (
        "Audit an AI-generated Token Bucket rate limiter (Google May 2026 Code Comprehension Benchmark); catch the AI using wall-clock time.time() instead of time.monotonic() and omitting threading.Lock / asyncio.Lock around shared token refill state.",
        "Write an empirical pytest proving clock drift vulnerability under mock leap seconds / NTP adjustments and a race condition under 50 concurrent async refill tasks."
    ),
    76: (
        "Audit an AI-generated graceful shutdown handler for an ASGI server; catch the AI dropping active SSE streams abruptly instead of sending terminal events and awaiting http.disconnect.",
        "Write a test client verifying that clients receive proper SSE termination without broken connection errors during server SIGTERM."
    ),
    83: (
        "Audit an AI-generated hybrid asyncio + multiprocessing worker pipeline; catch the AI using multiprocessing default fork instead of spawn alongside threads/event loops, or leaking SharedMemory segments when a child worker is terminated.",
        "Write a failure reproduction showing corrupted event loop state on fork and an OS shared memory leak verified via /dev/shm (or Windows named memory resource check)."
    ),
    90: (
        "Audit an AI-generated package entrypoint and API curation module; catch the AI omitting private submodules or creating cyclic imports in __init__.py that crash on clean import.",
        "Write an importlib isolation test proving circular import deadlocks under clean module initialization."
    ),
    97: (
        "Audit an AI-generated wheel packaging configuration and build script; catch the AI packaging temporary test fixtures, local caches, and mock secrets into the final distribution wheel.",
        "Write a wheel inspection script using zipfile verifying zero development credentials or test artifacts exist inside the built wheel."
    ),
    104: (
        "Audit an AI-generated chaos testing harness; catch the AI using hardcoded uncoordinated time.sleep() delays that produce non-deterministic flaky test failures.",
        "Write a test suite demonstrating flakiness under CPU contention, and replace sleep calls with deterministic async synchronization events."
    ),
    111: (
        "Audit an AI-generated refactoring of a production error-handling pipeline; catch the AI silently removing defensive type checks and exception branches because 'code coverage was low'.",
        "Write mutation tests showing that invalid payloads now crash with unhandled 500 errors instead of clean 422 validation diagnostics."
    ),
    118: (
        "Audit an AI-generated MiniModule parameter registration container; catch the AI returning shallow copies of parameters or failing to recurse into nested MiniModule submodules.",
        "Write a failing pytest proving that parameters of submodules are omitted from module.parameters(), and implement recursive generator traversal."
    ),
    125: (
        "Audit an AI-generated sys.monitoring execution tracer; catch the AI inserting profiler hooks that alter exception handling semantics or trigger infinite recursion on internal events.",
        "Write a test demonstrating stack overflow on recursive function inspection, and filter out profiler internal frames."
    ),
    132: (
        "Audit an AI-generated configuration hot-reloading mechanism; catch the AI performing non-atomic pointer updates during reload, exposing inconsistent configuration snapshots to concurrent reader threads.",
        "Write a multi-threaded consistency test reading config during continuous reloads, and fix using atomic pointer swaps or copy-on-write."
    ),
    139: (
        "Audit an AI-generated framework migration adapter; catch the AI swallowing ASGI exception scopes or losing chunk delimiters during streaming response translation.",
        "Write an ASGI test harness catching swallowed 500 exceptions and verifying chunk-delimited streaming integrity."
    ),
    146: (
        "Audit an AI-generated zero-copy ring buffer; catch the AI slicing memoryview into bytes on read, triggering hidden heap allocations on the hot path.",
        "Write a tracemalloc benchmark proving zero byte allocations during socket.recv_into buffer reads."
    ),
    153: (
        "Audit an AI-generated INT8/INT4 quantization kernel; catch the AI off-by-one bit-masking or signed/unsigned nibble unpacking bugs in INT4 group quantization.",
        "Write an empirical pytest proving quantized weight decoding diverges from reference float weights on negative nibble boundaries."
    ),
    160: (
        "Audit an AI-generated BPE tokenizer implementation; catch the AI corrupting multi-byte UTF-8 sequences when splitting raw bytes across token boundaries.",
        "Write an adversarial Unicode test verifying that emojis and non-ASCII scripts round-trip losslessly through encode and decode."
    ),
    167: (
        "Audit an AI-generated speculative decoding verification loop; catch the AI accepting draft tokens that disagree with target model greedy distribution under sampling temperature.",
        "Write a deterministic verification test asserting exact mathematical equivalence between speculative decode output and standard autoregressive decode."
    ),
    174: (
        "Audit an AI-generated numerical optimization for softmax; catch the AI applying unsafe micro-optimizations that violate IEEE-754 floating point associativity and cause NaN overflow.",
        "Write numerical stability tests comparing logits with large dynamic range (e.g. [-10000.0, 10000.0]) against reference torch.softmax to 1e-5 tolerance."
    ),
    181: (
        "Audit an AI-generated C FFI wrapper releasing the GIL; catch the AI accessing Python PyObject* heap pointers after Py_BEGIN_ALLOW_THREADS without re-acquiring the GIL.",
        "Write a multi-threaded stress test demonstrating CPython VM crash / segmentation fault, and fix by extracting raw C pointers before releasing GIL."
    ),
    188: (
        "Audit an AI-generated CUDA stream synchronization wrapper; catch the AI recording CUDA events on the wrong stream, causing host-device synchronization race conditions.",
        "Write an event synchronization test proving race conditions on out-of-order kernel completion, and associate events with the correct active stream."
    ),
    195: (
        "Audit an AI-generated zero-copy FFI stride calculation; catch the AI assuming contiguous C-memory layout when receiving strided slices from native buffers.",
        "Write a unit test passing a transposed non-contiguous buffer slice and proving corrupted memory read in the native extension."
    ),
    202: (
        "Audit an AI-generated distributed chaos testing script; catch the AI hardcoding network port numbers across distributed worker tests, causing port collision failures.",
        "Write a dynamic ephemeral port binding fixture and verify clean isolation across concurrent test runners."
    ),
    209: (
        "Audit an AI-generated Prometheus engine metrics collector; catch the AI computing Time-To-First-Token (TTFT) using wall-clock timestamps across different system clocks.",
        "Write an empirical test demonstrating negative TTFT anomalies under simulated clock skew, and convert all duration measurements to monotonic clock differences."
    ),
    216: (
        "Audit an AI-generated chaos post-mortem analyzer; catch the AI masking worker process OOM crash exit codes (SIGKILL / 137) as clean graceful shutdowns.",
        "Write a subprocess termination test asserting exit code 137 is correctly identified as an Out-Of-Memory failure with critical alert escalation."
    ),
    223: (
        "Audit an AI-generated curriculum bridge script; catch the AI generating unpinned environment dependencies or non-reproducible CUDA toolkit path assumptions.",
        "Write an environment verification test ensuring reproducible virtual environments across clean Linux/Windows systems."
    ),
}

COMPREHENSION_GATES = {
    35: (
        "week05_repo",
        "Cold-read an unfamiliar 350-line slotted sequence cache package. Squash 2 seeded bugs: (1) memory leak in __del__ cycle and (2) slice indexing off-by-one.",
        "Extend the sequence cache with dynamic capacity resizing without invalidating existing weakref keys.",
        "Profile memory footprint using sys.getsizeof; eliminate redundant dict allocations in cache entries."
    ),
    49: (
        "week07_repo",
        "Cold-read an unfamiliar 400-line priority scheduler and trie library. Squash 2 seeded bugs: (1) priority tie-breaker TypeError in heapq and (2) prefix trie infinite recursion on empty string key.",
        "Extend the scheduler with aging credit mechanisms preventing low-priority starvation under heavy queue pressure.",
        "Profile heap push/pop operations; refactor hot loop to avoid redundant tuple packing."
    ),
    63: (
        "week09_repo",
        "Cold-read an unfamiliar 450-line async task supervisor package. Squash 2 seeded bugs: (1) swallowed CancelledError preventing supervisor shutdown and (2) unreferenced background task GC drop.",
        "Implement a task health-probe heartbeat that restarts stalled worker tasks after a configurable timeout.",
        "Profile event loop latency; replace blocking file write with async chunked buffer."
    ),
    77: (
        "week11_repo",
        "Cold-read an unfamiliar 400-line raw ASGI streaming server. Squash 2 seeded bugs: (1) missing SSE end-of-frame double-newline and (2) slowloris socket timeout hanging forever.",
        "Implement streaming request body chunking with a strict max_body_bytes defense.",
        "Profile socket send_into throughput; eliminate intermediate string allocations in SSE serialization."
    ),
    91: (
        "week13_repo",
        "Cold-read an unfamiliar 450-line plugin registry and dependency container. Squash 2 seeded bugs: (1) circular dependency undetected causing RecursionError and (2) duplicate plugin registration overwriting handlers.",
        "Implement a scoped lifecycle manager supporting singleton vs transient service resolution.",
        "Profile dependency graph resolution; cache resolved topological order."
    ),
    119: (
        "week17_repo",
        "Cold-read an unfamiliar 400-line declarative schema compiler and MiniModule tree. Squash 2 seeded bugs: (1) submodule parameter omission in recursive traversal and (2) __init_subclass__ keyword argument drop.",
        "Extend MiniModule with a named_parameters() generator supporting regex filtering by parameter path.",
        "Profile module registration overhead; eliminate redundant parameter dictionary cloning."
    ),
    133: (
        "week19_repo",
        "Cold-read an unfamiliar 450-line dynamic route dispatcher and middleware chain. Squash 2 seeded bugs: (1) path parameter regex collision matching incorrect route and (2) middleware exception leaving response unclosed.",
        "Implement route-level rate limiting middleware with an in-memory sliding window.",
        "Profile route matching; compile unified regex tree replacing linear route iteration."
    ),
    147: (
        "week21_repo",
        "Cold-read an unfamiliar 450-line zero-copy memory slab allocator and ring buffer. Squash 2 seeded bugs: (1) memoryview slice out-of-bounds on non-aligned offset and (2) slab reuse before refcount reaches zero.",
        "Extend the allocator with a slab defragmentation compaction routine.",
        "Profile allocation latency; eliminate heap allocations in the hot buffer acquisition path."
    ),
    161: (
        "week23_repo",
        "Cold-read an unfamiliar 500-line pure-Python Transformer execution graph. Squash 2 seeded bugs: (1) RoPE rotation applied to values instead of Q/K and (2) KV-cache index overwrite during autoregressive decode.",
        "Implement top-k / top-p temperature sampling inside the token generation loop.",
        "Profile attention matrix multiplication; optimize inner loop using strided slice views."
    ),
    175: (
        "week25_repo",
        "Cold-read an unfamiliar 450-line stack-sampling flamegraph profiler. Squash 2 seeded bugs: (1) profiler frame sampling dropping C-extension frames and (2) wall-clock time skew during sample intervals.",
        "Implement an allocation hotspot analyzer reporting top 10 allocation call stacks with byte totals.",
        "Profile the profiler overhead; reduce sampling hook overhead to < 3% execution slowdown."
    ),
    189: (
        "week27_repo",
        "Cold-read an unfamiliar 450-line native C/CUDA FFI bridge and MetaTensor library. Squash 2 seeded bugs: (1) ctypes argtypes pointer mismatch causing segfault and (2) premature GC of underlying memoryview buffer.",
        "Implement a zero-copy DLPack capsule export allowing seamless tensor sharing with external runtimes.",
        "Profile FFI transition overhead; batch native kernel invocations to minimize ctypes dispatch cost."
    ),
    203: (
        "week29_repo",
        "Cold-read an unfamiliar 500-line distributed tensor-parallel IPC worker group. Squash 2 seeded bugs: (1) all-reduce reduction sum missing rank scaling and (2) barrier deadlock on single worker timeout.",
        "Implement prefix-aware request routing forwarding prompts with common prefixes to the same worker rank.",
        "Profile inter-process communication latency; replace pickle serialization with raw struct packing."
    ),
}

DAY_CUSTOMIZATIONS = {
    1: [
        "- [ ] **[CORE] Task 1: Setup, Reading & Diagnostic Error Literacy Walkthrough (12m + 3m):** Open `py-mastery/`, check `git status`, create `artifacts/day001/`, and review the mapped book section. Walk through the diagnostic guide on reading `mypy --strict` invariance/narrowing errors (e.g. `Incompatible types in assignment` and variance mismatches in `List[Derived]` vs `List[Base]`) and decoding `pytest` failure diffs (`-` vs `+`, assertion rewrite inspection) so static and runtime errors can be systematically decoded from Day 1.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (20m):** Create `py-mastery/` with `pyproject.toml`, `src/pymastery/__init__.py`, and `src/pymastery/env_info.py` that inspects interpreter version, platform, and active prefix cleanly with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` unit and edge-case tests verifying environment inspection and save the test log beside `pyproject.toml`, `src/pymastery/env_info.py`, and `tests/test_env_info.py`.",
        "- [ ] **[MINOR] Task 4: Diagnostic Literacy Practice & Static Audit (11m):** Run `mypy --strict src tests` and `ruff check src tests`. Practice decoding three pre-seeded mypy invariance errors and record diagnostic triage rules in `artifacts/day001/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately introduce an invalid type assignment in `env_info.py`, observe how `mypy --strict` pinpoints the invariance mismatch, restore the fix, and log the diagnostic trace in `artifacts/day001/failure-notes.md`."
    ],
    19: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day019/`, and review Fluent Python Ch. 8 & pytest documentation.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (22m):** Implement test fixtures, `tmp_path`, `monkeypatch`, `caplog`, and parametrized test suites for a token-bucket rate limiter module with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests & AI Test Audit (10m):** Receive a 10-test AI-generated pytest suite for the rate limiter that passes 100% despite seeded defects. Audit the test suite to identify tautological assertions that assert buggy behavior rather than system invariants.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (10m):** Static typing (`mypy --strict`) and linter (`ruff`) enforcement: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day019/quality.log`.",
        "- [ ] **Task 5: Auditing Tautological Tests (8m):** Rewrite the AI test suite to eliminate tautological assertions, assert correct rate-limiting invariants, and verify the rewritten suite kills all 3 seeded implementation mutants in `drills/day019_mutants.py`, logging results in `artifacts/day019/failure-notes.md`."
    ],
    20: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day020/`, and review test double architecture patterns.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (22m):** Implement stateful test doubles (`FakeClock`, `FakeMetricsSink`) from scratch without third-party mocking libraries (`unittest.mock`) with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Test Doubles vs Over-Mocking Audit (10m):** Audit an AI-generated test suite that over-mocks internal methods using `unittest.mock.patch`, observing how tests pass even when private method signatures change.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (10m):** Static typing (`mypy --strict`) and linter (`ruff`) enforcement: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day020/quality.log`.",
        "- [ ] **Task 5: Replacing Brittle Mocks with Fakes (8m):** Replace brittle `unittest.mock.patch` calls with stateful `FakeClock` and `FakeMetricsSink`. Prove the fake-based tests remain resilient to internal refactoring while killing 3 seeded timing mutants in `drills/day020_buggy.py` within 8 minutes."
    ],
    21: [
        "- [ ] **[CORE] Task 1: Environment Setup & Bisect Branch Preparation (10m):** Open `py-mastery/`, prepare git repository with a 10-commit history branch containing a seeded resource leak regression, and review Week 3 consolidation criteria.",
        "- [ ] **[CORE] Task 2: From-Scratch Gate Implementation (22m):** Consolidate Week 3 fault-tolerant resource management and test doubles into a robust service component with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Comprehensive Gate Test Suite (10m):** Write regression and chaos tests verifying resource clean-up and save the test log beside `gates/week03-gate.md`.",
        "- [ ] **[MINOR] Task 4: Git Bisect Archaeology & Packet Narration (11m):** Execute `git bisect run pytest` across the 10-commit history branch to automatically isolate the exact commit SHA that introduced the resource leak. Then dedicate 3 minutes to speaking out loud for 2 minutes and recording the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week03-gate.md`: (1) Clarifying Assumptions, (2) Pattern/Structure Chosen & Why, (3) Time & Space Complexity Bound, (4) Failure Mode Under 10x Load or Malicious Input, and (5) Primary Design Trade-Off Accepted.",
        "- [ ] **Task 5: Failure Diagnosis & Clean-Room Verification (7m):** Verify that the isolated regression commit is documented, the fix passes all static and unit test gates, and record verification proof in `artifacts/day021/failure-notes.md`."
    ],
    28: [
        "- [ ] **[CORE] Task 1: Clean Environment & Baseline Test Run (10m):** Open `py-mastery/`, verify baseline test suite passes, create `artifacts/day028/`, and review Month 1 systems architecture boundaries.",
        "- [ ] **[CORE] Task 2: From-Scratch Gate Implementation (22m):** Consolidate Month 1 packaging, strict typing, error trees, context managers, and lazy streaming pipelines into `src/pymastery/` with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Comprehensive Test Suite (10m):** Run full test suite with coverage reporting, asserting > 90% branch coverage across all Month 1 modules.",
        "- [ ] **[MINOR] Task 4: PR Diff Code Review Critique & Packet Narration (11m):** Review a 150-line AI-generated pull-request diff containing 5+ line duplicated code blocks (GitClear churn pattern), leaky abstractions, and subtle bugs. Write structured review comments in `gates/week04-pr-review.md`. Then dedicate 3 minutes to speaking out loud for 2 minutes and recording the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week04-gate.md`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day028_buggy.py` (PR diff churn and leaky abstraction), formulate a 1-sentence root-cause hypothesis in `artifacts/day028/failure-notes.md`, fix the defect, and verify clean regression pass within 7 minutes."
    ],
    44: [
        "- [ ] **[CORE] Task 1: Setup & Reading (8m):** Open `py-mastery/`, check `git status`, create `artifacts/day044/`, and review priority queue structures in Fluent Python Ch. 2.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (24m):** Implement `FairRequestScheduler` in `src/pymastery/scheduler.py` using `heapq` with custom priority tie-breaking with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (10m):** Write `pytest` unit and edge-case tests verifying priority ordering and tie-breaking, saving test log beside `src/pymastery/scheduler.py`.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (6m):** Static typing (`mypy --strict`) and linter (`ruff`) enforcement: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day044/quality.log`.",
        "- [ ] **Task 5: Constraint-Shift Follow-Up Drill (12m):** Shift constraint to match Anthropic's credit-based GPU scheduler prompt: prevent low-priority request starvation by adding monotonic credit aging where waiting requests gain priority credits over time. Verify with a starvation-prevention test and log in `artifacts/day044/failure-notes.md`."
    ],
    46: [
        "- [ ] **[CORE] Task 1: Setup & Reading (8m):** Open `py-mastery/`, check `git status`, create `artifacts/day046/`, and review doubly linked list and hash map caching patterns.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (24m):** Implement an O(1) `LRUCache` from scratch in `src/pymastery/lru.py` using Doubly Linked List nodes and Dict with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (10m):** Write `pytest` tests verifying O(1) eviction, `get`, `put`, and capacity limits.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (6m):** Memory footprint and benchmark verification: Run memory profile and Big-O verification; record zero errors in `artifacts/day046/quality.log`.",
        "- [ ] **Task 5: Constraint-Shift Follow-Up Drill (12m):** Shift constraint to match Anthropic's LRU follow-up: convert from item-count capacity to byte-size-weighted capacity (`max_bytes`) + monotonic TTL expiration. Introduce a simple randomized reference-oracle fuzz test verifying cache state against a naive dictionary oracle, logging results in `artifacts/day046/failure-notes.md`."
    ],
    47: [
        "- [ ] **[CORE] Task 1: Setup & Reading (8m):** Open `py-mastery/`, check `git status`, create `artifacts/day047/`, and review Trie and Radix prefix trees for KV-cache prefix sharing.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (24m):** Implement `TokenPrefixTrie` in `src/pymastery/trie.py` for longest prefix matching and token sequence indexing with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (10m):** Write `pytest` tests verifying longest common prefix lookup, insert, and branch pruning.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (6m):** Complexity and performance audit: Benchmark trie lookup latency vs list scanning; record zero errors in `artifacts/day047/quality.log`.",
        "- [ ] **Task 5: Constraint-Shift Follow-Up Drill (12m):** Shift constraint to support reference-counted prefix blocks (`ref_count > 0`) that cannot be evicted while active inference requests are decoding from them. Write a test asserting that attempting to prune active nodes raises a lock exception, logging results in `artifacts/day047/failure-notes.md`."
    ],
    51: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day051/`, and review `pathlib` and defensive filesystem I/O.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation & Security Hardening (25m):** Implement `AtomicJsonStore` in `src/pymastery/atomic_store.py` with atomic write-and-rename semantics and adversarial path sanitization: enforce `Path.resolve()` containment checks rejecting path-traversal (`../`) and symlink-escape attacks with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Security & Correctness Unit Tests (7m):** Write `pytest` unit tests verifying atomic write guarantees alongside adversarial security tests asserting rejection of directory traversal payloads.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Defensive filesystem and serialization audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day051/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately attempt a symlink-escape write, verify that the security check aborts with `SecurityError`, and log the attack trace in `artifacts/day051/failure-notes.md`."
    ],
    52: [
        "- [ ] **[CORE] Task 1: Setup, Reading & Conceptual Zero-Copy Primer (12m + 3m):** Open `py-mastery/`, check `git status`, create `artifacts/day052/`, and review SafeTensors binary layout. Complete a 10-minute conceptual primer bridging `mmap` and `memoryview` zero-copy slicing ahead of Month 6, and inspect the 10-line header parsing trap snippet.",
        "- [ ] **[CORE] Task 2: From-Scratch SafeTensors Parser with Adversarial Validation (20m):** Implement `src/pymastery/safetensors.py` to parse 64-bit uint64 header lengths and JSON metadata with strict security bounds: reject header-length bombs (> 100 MB) and non-little-endian length fields with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit & Adversarial Tests (7m):** Write `pytest` tests validating valid SafeTensors headers and asserting rejection of malformed or explosive 64-bit header lengths.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Binary data parsing and bounds audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day052/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day052_buggy.py` (SafeTensors 64-bit JSON header length parsed with system endianness instead of little-endian uint64), formulate a hypothesis in `artifacts/day052/failure-notes.md`, and fix the defect within 7 minutes."
    ],
    53: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day053/`, and review memory-mapped files and `memoryview` zero-copy slicing.",
        "- [ ] **[CORE] Task 2: From-Scratch mmap Binary Reader with Offset Hardening (25m):** Implement `src/pymastery/mmap_reader.py` using `mmap` and `memoryview` to extract tensor slices with zero memory copies, adding adversarial validation that rejects overlapping or out-of-bounds tensor `data_offsets` with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Memory & Adversarial Unit Tests (7m):** Write `pytest` and `tracemalloc` tests proving zero heap allocations during tensor slicing and asserting rejection of overlapping tensor buffer offsets.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Zero-copy memory and resource audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day053/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately craft a SafeTensors file with overlapping tensor byte offsets, verify that the reader rejects it with `ValueError: Overlapping tensor buffers detected`, and record in `artifacts/day053/failure-notes.md`."
    ],
    56: [
        "- [ ] **[CORE] Task 1: Level 1 — Core In-Memory Container & Type Contracts (12m):** Implement the foundational in-memory caching container with strict types (`Protocol`, frozen dataclasses) under a 4-level progressive refactoring interview simulation with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 2: Level 2 — Prefix/Range Filtering & Sorting (15m):** Unlock Level 2 upon passing Level 1 tests: implement prefix and range filtering with sorted key retrieval without ground-up rewrites.",
        "- [ ] **[CORE] Task 3: Level 3 — Monotonic TTL Expiration & Atomic State Rollback (15m):** Unlock Level 3 upon passing Level 2 tests: implement monotonic TTL eviction and transactional state rollback under write failure.",
        "- [ ] **[MINOR] Task 4: Level 4 — Thread/Async Safe Compaction & PR Diff Critique (11m):** Unlock Level 4: implement thread-safe batch compaction and binary snapshot serialization. Perform a PR diff code review critique on a 150-line AI-generated refactoring diff, identifying 5+ line duplicated blocks and leaky abstractions in `gates/week08-pr-review.md`.",
        "- [ ] **Task 5: Hiring Committee Packet Narration & Clean-Room Verification (7m):** Speak out loud for 2 minutes and record the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week08-gate.md`: (1) Clarifying Assumptions, (2) Pattern/Structure Chosen & Why, (3) Time & Space Complexity Bound, (4) Failure Mode Under 10x Load or Malicious Input, and (5) Primary Design Trade-Off Accepted."
    ],
    68: [
        "- [ ] **[CORE] Task 1: Setup & Reading (8m):** Open `py-mastery/`, check `git status`, create `artifacts/day068/`, and review dynamic micro-batching patterns with Python Concurrency with asyncio Ch. 5.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (24m):** Implement `AsyncDynamicBatcher` in `src/pymastery/batcher.py` using per-request `asyncio.Future` and an async batch collection loop with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (10m):** Write `pytest` tests verifying batch size thresholds, max delay timeouts, and concurrent client completions.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (6m):** Concurrency and latency audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day068/quality.log`.",
        "- [ ] **Task 5: Constraint-Shift Follow-Up Drill (12m):** Shift constraint to handle caller timeout cancellation: when a caller abandons a request that is already queued or batched, cancel the caller without raising `asyncio.InvalidStateError` when the batch worker eventually resolves the Future. Verify with 100 concurrent requests and log in `artifacts/day068/failure-notes.md`."
    ],
    69: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day069/`, and review token bucket rate limiting algorithms and monotonic time.",
        "- [ ] **[CORE] Task 2: From-Scratch Implementation (25m):** Implement `TokenBucketRateLimiter` in `src/pymastery/rate_limiter.py` with monotonic clock refill, thread/async safety, and per-tenant burst capacity with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` tests validating burst allowances, steady-state refill rates, and multi-tenant isolation.",
        "- [ ] **[MINOR] Task 4: Mode B Constrained AI Audit (10m):** Practice the Google May 2026 Code Comprehension Benchmark: Prompt an AI assistant to generate a Token Bucket rate limiter. Audit the AI output for classic flaws: using wall-clock `time.time()` instead of `time.monotonic()` and omitting `threading.Lock` / `asyncio.Lock` around shared token refill state.",
        "- [ ] **Task 5: Mode B Hallucination Proof & Artifact Logging (8m):** Write an empirical `pytest` proving clock drift vulnerability under mock leap seconds / NTP adjustments and race conditions under 50 concurrent async refill tasks. Fix the defects and log prompt and proofs in `artifacts/day069/ai-audit.md`."
    ],
    73: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day073/`, and review HTTP/1.1 wire framing and Server-Sent Events (`text/event-stream`) specification.",
        "- [ ] **[CORE] Task 2: From-Scratch SSEParser with Buffer Hardening (25m):** Implement `SSEParser` in `src/pymastery/sse.py` parsing event streams with event names, IDs, multi-line data, and a strict `max_line_bytes` buffer cap to prevent unbounded memory-exhaustion DoS on newline-free malicious streams with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Streaming & Security Unit Tests (7m):** Write `pytest` tests for SSE event framing, reconnect IDs, and assert immediate rejection of lines exceeding `max_line_bytes`.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Streaming protocol compliance audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day073/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Feed an infinite newline-free byte stream into the parser, verify that `max_line_bytes` triggers a `BufferOverflowError` without exhausting heap memory, and record in `artifacts/day073/failure-notes.md`."
    ],
    75: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day075/`, and review the ASGI specification (`scope`, `receive`, `send`).",
        "- [ ] **[CORE] Task 2: From-Scratch Raw ASGI Server with Slowloris Defense (25m):** Implement a zero-dependency raw ASGI HTTP server in `src/pymastery/asgi.py` with maximum request body byte limits (`max_body_bytes`) and header read timeouts defending against Slowloris denial-of-service attacks with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated ASGI & Security Tests (7m):** Write tests verifying HTTP lifespan, request-response cycles, and timeout termination on slow partial header streams.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** ASGI protocol compliance audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day075/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Simulate a slow-sending HTTP client that trickles 1 byte per second, verify that the header read timeout disconnects the client within configured deadline, and log in `artifacts/day075/failure-notes.md`."
    ],
    84: [
        "- [ ] **[CORE] Task 1: Multi-Process Baseline & Bisect Setup (10m):** Open `py-mastery/`, prepare a 10-commit history branch containing an introduced latency regression, and review Month 3 systems integration.",
        "- [ ] **[CORE] Task 2: From-Scratch Service Implementation (22m):** Build `src/pymastery/multi_process_service.py` integrating async frontend routing with multiprocessing SharedMemory workers with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Concurrent End-to-End Test Suite (10m):** Write stress tests sending 100 concurrent requests across worker processes.",
        "- [ ] **[MINOR] Task 4: Git Bisect Latency Archaeology & Packet Narration (11m):** Execute `git bisect run pytest` across the 10-commit history branch to automatically isolate the exact commit SHA that introduced the latency regression. Then dedicate 3 minutes to speaking out loud for 2 minutes and recording the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week12-gate.md`.",
        "- [ ] **Task 5: Failure Diagnosis & Clean-Room Verification (7m):** Verify that the regression is resolved, workers clean up SharedMemory on abnormal exit, and record verification proof in `artifacts/day084/failure-notes.md`."
    ],
    89: [
        "- [ ] **[CORE] Task 1: Setup & Legacy Code Review (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day089/`, and review legacy refactoring patterns.",
        "- [ ] **[CORE] Task 2: From-Scratch Refactoring (22m):** Refactor messy, untyped legacy code in `drills/legacy_module.py` into clean, typed, modular code in `src/pymastery/refactored.py` with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Characterization Test Suite (10m):** Write golden characterization tests ensuring zero behavioral drift between legacy and refactored versions.",
        "- [ ] **[MINOR] Task 4: PR Diff Code Review Critique (11m):** Review a 150-line AI-generated pull-request diff containing 5+ line duplicated code blocks (GitClear churn pattern), leaky abstractions, and subtle regressions. Document structured code review comments in `artifacts/day089/pr-review.md`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle regression in `drills/day089_buggy.py`, write a 1-sentence root-cause hypothesis in `artifacts/day089/failure-notes.md`, and fix the bug within 7 minutes."
    ],
    94: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day094/`, and review `tomllib`, configuration parsing, and timing side-channel attacks.",
        "- [ ] **[CORE] Task 2: From-Scratch Config Loader with Timing Defense (25m):** Implement `src/pymastery/config.py` loading TOML configurations with automatic environment variable overrides, `SecretStr` string masking in `__repr__`, and constant-time token comparison via `hmac.compare_digest` to prevent timing side-channel attacks with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Security & Config Tests (7m):** Write tests verifying secret masking in logs/tracebacks and constant-time authentication token validation.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Security and configuration hygiene audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day094/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Verify that standard `==` string equality on secrets fails security lint checks, restore `hmac.compare_digest`, and log the timing defense rationale in `artifacts/day094/failure-notes.md`."
    ],
    109: [
        "- [ ] **[CORE] Task 1: Level 1 — Core In-Memory Container & Type Contracts (12m):** Implement the foundational schema repository and in-memory container under an Anthropic 4-Level Progressive Coding simulation with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 2: Level 2 — Prefix/Range Filtering & Sorting (15m):** Unlock Level 2 upon passing Level 1 tests: implement prefix and range queries with sorted index ordering without rewriting the core container.",
        "- [ ] **[CORE] Task 3: Level 3 — Monotonic TTL Expiration & Atomic State Rollback (15m):** Unlock Level 3 upon passing Level 2 tests: implement monotonic TTL eviction and transactional state rollback under failed insertions.",
        "- [ ] **[MINOR] Task 4: Level 4 — Thread/Async Safe Batch Compaction & Serialization (11m):** Unlock Level 4: implement thread-safe batch compaction and binary snapshot serialization. Verify static typing with `mypy --strict` and `ruff check`.",
        "- [ ] **Task 5: Failure Diagnosis & Progressive Benchmark (7m):** Pass all 4 escalating levels within the 60-minute contract, verify that earlier level tests remain unbroken, and record architectural trade-offs in `artifacts/day109/progressive-refactor.md`."
    ],
    120: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day120/`, and review runtime signature inspection in Fluent Python Ch. 7, 9.",
        "- [ ] **[CORE] Task 2: From-Scratch Function Introspection & Tool-Call Sandbox Foundation (25m):** Implement `src/pymastery/inspector.py` using `inspect.signature` to introspect arbitrary Python callables, generating strict JSON Schema specifications and establishing the foundations for an Agentic Tool-Call Execution Sandbox with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` tests validating JSON schema extraction across positional, keyword, and typed default arguments.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Dynamic inspection and typing audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day120/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day120_buggy.py` (signature introspection failing on decorated functions without `__wrapped__` unwrapping), formulate a hypothesis in `artifacts/day120/failure-notes.md`, and fix within 7 minutes."
    ],
    136: [
        "- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day136/`, and review HTTP routing architectures and SSRF vulnerability patterns.",
        "- [ ] **[CORE] Task 2: From-Scratch Route Dispatcher with SSRF Defense (25m):** Implement `NanoServe` route registration (`@app.post`), path parameter parsing, and HTTP request handling with strict payload size limits and URL validation rejecting SSRF targets (loopback and cloud metadata IPs) with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Routing & Security Unit Tests (7m):** Write `pytest` tests for parameter extraction and assert immediate rejection of SSRF URL destinations.",
        "- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** Web framework and routing audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day136/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Attempt to pass a cloud metadata URL (`http://169.254.169.254/latest/meta-data/`) to an outbound handler, verify that SSRF validation raises `SecurityError`, and log in `artifacts/day136/failure-notes.md`."
    ],
    180: [
        "- [ ] **[CORE] Task 1: Setup & Reference Template Inspection (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day180/`, and inspect the provided reference template `fast_ops.c` containing 2 seeded pointer-lifetime and 64-byte buffer-alignment bugs.",
        "- [ ] **[CORE] Task 2: Audit C Boundary, Fix Pointer Bugs & Bind via ctypes (25m):** Audit the C boundary in `fast_ops.c`, fix the pointer lifetime and alignment defects, declare strict `ctypes` `argtypes` and `restype`, pin Python `memoryview` buffer lifetimes against premature GC, and release the GIL with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Numerical Parity & Alignment Tests (7m):** Write `pytest` unit tests verifying numerical parity to `1e-5` between C kernel and pure-Python reference implementation under strided inputs.",
        "- [ ] **[MINOR] Task 4: AddressSanitizer / Memory Audit (11m):** AddressSanitizer and valgrind memory audit: Compile with `-fsanitize=address` and verify zero out-of-bounds reads or memory leaks; record zero errors in `artifacts/day180/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day180_buggy.py` (C99 fast_ops.c buffer-alignment and pointer-lifetime boundary defect), formulate a hypothesis in `artifacts/day180/failure-notes.md`, and fix the defect within 7 minutes."
    ],
    184: [
        "- [ ] **[CORE] Task 1: Setup & C++ Template Inspection (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day184/`, and inspect the provided reference template `paged_attention_binding.cpp` containing 2 seeded pointer-lifetime and buffer-alignment bugs.",
        "- [ ] **[CORE] Task 2: Audit C++ Boundary, Fix GC Lifetimes & Declare Nanobind Exports (25m):** Audit the C++ boundary, fix pointer lifetime and alignment defects, declare strict nanobind/pybind11 module exports, pin Python `memoryview` buffer lifetimes against premature GC, and release the GIL during kernel execution with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Numerical & Memory Stress Tests (7m):** Write `pytest` unit tests verifying paged KV-cache gathering correctness and memory stability under 1,000 continuous gathering iterations.",
        "- [ ] **[MINOR] Task 4: Memory Leak & FFI Audit (11m):** Native FFI memory leak audit: Verify zero leaked Python heap references across 10,000 native calls; record zero errors in `artifacts/day184/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day184_buggy.py` (C++ paged_attention_binding.cpp memoryview lifetime premature GC collection defect), formulate a hypothesis in `artifacts/day184/failure-notes.md`, and fix within 7 minutes."
    ],
    204: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM Engine (10m):** Clone the official `vllm` repository alongside `py-mastery/`. Use Mode B constrained prompts to trace `vllm/v1/engine/async_llm.py`, `vllm/v1/engine/core.py`, and `vllm/v1/core/sched/scheduler.py`, identifying how `AsyncLLM` queues requests to `EngineCore`.",
        "- [ ] **[CORE] Task 2: From-Scratch Clean-Room Implementation in vllm_mini (22m):** Implement `AsyncLLM` and `EngineCore` separation in `src/pymastery/vllm_mini/engine.py` with asynchronous request submission, task queueing, and background engine loop with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit Tests (10m):** Write `pytest` unit tests verifying async request submission, request ID generation, and output stream generation.",
        "- [ ] **[MINOR] Task 4: Static Audit & Architectural Parity Verification (11m):** Static typing (`mypy --strict`) and linter (`ruff`) enforcement: Run `mypy --strict src tests` and verify architectural parity with vLLM v1 core; record zero errors in `artifacts/day204/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day204_buggy.py` (vLLM async LLM engine request queue drop under burst arrival), formulate a hypothesis in `artifacts/day204/failure-notes.md`, and fix within 7 minutes."
    ],
    205: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM Detokenizer (10m):** Trace `vllm/transformers_utils/detokenizer.py` in the cloned vLLM repo using Mode B constrained prompts to understand incremental UTF-8 subword holdback and prefix detokenization.",
        "- [ ] **[CORE] Task 2: From-Scratch IncrementalDetokenizer Implementation (22m):** Implement `IncrementalDetokenizer` in `src/pymastery/vllm_mini/detokenizer.py` buffering incomplete multi-byte UTF-8 sequences and emitting text incrementally with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Unit & Unicode Tests (10m):** Write `pytest` tests validating partial multi-byte UTF-8 character holdback (e.g. 3-byte and 4-byte emoji sequences) and prompt-prefix token stripping.",
        "- [ ] **[MINOR] Task 4: Static Audit & Unicode Fuzzing (11m):** Unicode boundary and memory audit: Fuzz detokenizer with randomized byte slices; record zero errors in `artifacts/day205/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately emit incomplete UTF-8 bytes to verify immediate holdback without replacement character corruption, restoring fix and logging in `artifacts/day205/failure-notes.md`."
    ],
    206: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM Serving Protocol (10m):** Trace `vllm/entrypoints/openai/serving_chat.py` and `protocol.py` in the cloned vLLM repo using Mode B constrained prompts to map OpenAI chat completion schema contracts.",
        "- [ ] **[CORE] Task 2: From-Scratch Protocol Implementation in vllm_mini (22m):** Implement OpenAI-compatible `/v1/chat/completions` protocol models and SSE chunk streaming framing in `src/pymastery/vllm_mini/protocol.py` with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Protocol Validation Tests (10m):** Write `pytest` tests validating request serialization, streaming delta chunks, finish reasons, and terminal `[DONE]` markers.",
        "- [ ] **[MINOR] Task 4: Static Audit & OpenAPI Spec Check (11m):** API protocol compliance audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day206/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day206_buggy.py` (OpenAI /v1/chat/completions protocol chunk framing missing terminal [DONE] SSE marker), formulate a hypothesis in `artifacts/day206/failure-notes.md`, and fix within 7 minutes."
    ],
    207: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM Guided Decoding (10m):** Trace `vllm/model_executor/guided_decoding/` (FSM logits processors) in the cloned vLLM repository using Mode B constrained prompts to analyze JSON grammar transitions.",
        "- [ ] **[CORE] Task 2: From-Scratch Guided Masking & Tool-Call Sandbox Harness (22m):** Implement guided JSON logits masking in `src/pymastery/vllm_mini/guided.py`, connecting Day 120's signature introspection to wire a deterministic Tool-Call Execution Sandbox & Evaluation Harness that validates emitted JSON tool arguments via NanoSchema, executes the tool under an `asyncio.timeout` boundary, and logs groundedness and latency metrics with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Grammar & Sandbox Tests (10m):** Write tests verifying logits masking against a JSON schema and asserting deterministic tool execution under timeout boundaries.",
        "- [ ] **[MINOR] Task 4: Static Audit & Evaluation Metrics Check (11m):** Schema enforcement and grounding audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day207/quality.log`.",
        "- [ ] **Task 5: Failure & Diagnosis Test (7m):** Feed malformed JSON arguments to the tool sandbox, verify that validation fails cleanly without tool execution, and log metrics in `artifacts/day207/failure-notes.md`."
    ],
    208: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM LoRA Manager (10m):** Trace `vllm/lora/worker_manager.py` and `models.py` in the cloned vLLM repo using Mode B constrained prompts to examine LRU adapter slot allocation and dynamic weight swapping.",
        "- [ ] **[CORE] Task 2: From-Scratch LoRACacheManager Implementation (22m):** Implement `LoRACacheManager` in `src/pymastery/vllm_mini/lora.py` with multi-adapter LRU caching, active slot pinning, and dynamic weight delta application with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Multi-Adapter Tests (10m):** Write `pytest` tests validating concurrent multi-adapter inference requests, cache hit scoring, and LRU eviction under memory caps.",
        "- [ ] **[MINOR] Task 4: Static Audit & Slot Concurrency Check (11m):** LoRA adapter lifecycle audit: Run `mypy --strict src tests` and `ruff check src tests`; record zero errors in `artifacts/day208/quality.log`.",
        "- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the subtle defect in `drills/day208_buggy.py` (LoRA multi-adapter LRU cache manager evicting active adapter while in-flight decode step runs), formulate a hypothesis in `artifacts/day208/failure-notes.md`, and fix within 7 minutes."
    ],
    209: [
        "- [ ] **[CORE] Task 1: Clone & Trace Open-Source vLLM Engine Metrics (10m):** Trace `vllm/engine/metrics.py` in the cloned vLLM repo using Mode B constrained prompts to examine Prometheus gauges, counters, and TTFT/ITL histogram buckets.",
        "- [ ] **[CORE] Task 2: From-Scratch Engine Observability Hook Implementation (22m):** Implement `EngineObservabilityHook` in `src/pymastery/vllm_mini/metrics.py` tracking real-time TTFT, ITL, KV-cache usage %, preemption count, and goodput with AI autocomplete strictly OFF.",
        "- [ ] **[CORE] Task 3: Automated Observability & Histogram Tests (10m):** Write `pytest` tests verifying metric updates, histogram bucket accumulations, and Prometheus exposition formatting.",
        "- [ ] **[MINOR] Task 4: Mode B Constrained AI Audit (10m):** Prompt an AI assistant to implement a Prometheus engine metrics collector. Audit the AI output for subtle timing hallucinations: catch the AI computing Time-To-First-Token (TTFT) using wall-clock `time.time()` instead of `time.monotonic()` across distributed server nodes.",
        "- [ ] **Task 5: Mode B Hallucination Proof & Artifact Logging (8m):** Write an empirical test demonstrating negative TTFT anomalies under simulated clock skew, convert all duration measurements to monotonic clock differences, and record in `artifacts/day209/ai-audit.md`."
    ]
}

def generate_day_tasks(
    day_num: int,
    week_num: int,
    month_num: int,
    title: str,
    objective: str,
    practical: str,
    deliverable: str,
    stop_cond: str,
    dont_study: str,
    minor_focus: str,
    minor_cmd: str,
    trap_warning: str,
):
    if day_num in DAY_CUSTOMIZATIONS:
        return DAY_CUSTOMIZATIONS[day_num]

    # Category A: 12 Unfamiliar Codebase Comprehension Simulation Gates
    if day_num in COMPREHENSION_GATES:
        repo_name, cold_desc, feature_desc, opt_desc = COMPREHENSION_GATES[day_num]
        return [
            f"- [ ] **[CORE] Task 1: Phase 1 — Cold Code Exploration & Call-Graph Tracing (10m):** Cold-read the unfamiliar 300–500 line multi-file codebase in `comprehension_labs/week{week_num:02d}_repo/`. Map execution flow, identify module boundaries, and form independent hypotheses before touching code.",
            f"- [ ] **[CORE] Task 2: Phase 1 — Diagnostic Bug Squash (10m):** {cold_desc} Verify fixes pass regression tests without breaking existing public contracts.",
            f"- [ ] **[CORE] Task 3: Phase 2 — Architectural Feature Extension (22m):** {feature_desc} Adhere strictly to existing architectural abstractions, types, and styling with AI autocomplete strictly OFF.",
            f"- [ ] **[MINOR] Task 4: Phase 3 — Bottleneck Profiling & Static Audit (10m):** {opt_desc} Verify with `mypy --strict` and `ruff check`, recording zero errors in `gates/week{week_num:02d}-quality.log`.",
            f"- [ ] **Task 5: Phase 3 — Hiring Committee Packet Narration (8m):** Speak out loud for 2 minutes and record the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week{week_num:02d}-gate.md`: (1) Clarifying Assumptions, (2) Pattern/Structure Chosen & Why, (3) Time & Space Complexity Bound, (4) Failure Mode Under 10x Load or Malicious Input, and (5) Primary Design Trade-Off Accepted."
        ]

    # Category B: Weekly Gates (other 20 gate days)
    if day_num % 7 == 0:
        return [
            f"- [ ] **[CORE] Task 1: Clean Environment & Baseline Test Run (10m):** Open `py-mastery/`, verify baseline environment cleanly, create `artifacts/day{day_num:03d}/`, and review Week {week_num} gate criteria for `{title}`.",
            f"- [ ] **[CORE] Task 2: From-Scratch Gate Implementation (22m):** {practical} Type every class and signature from a blank buffer with AI autocomplete strictly OFF.",
            f"- [ ] **[CORE] Task 3: Comprehensive Gate Test Suite (10m):** Write and run `pytest` unit and chaos tests verifying `{title}` and save the test log beside {deliverable}.",
            f"- [ ] **[MINOR] Task 4: Quality Audit & Hiring Committee Packet Narration (11m):** {minor_focus}: {minor_cmd}. Then dedicate 3 minutes to speaking out loud for 2 minutes and recording the mandatory 5-bullet Hiring Committee Packet Summary in `gates/week{week_num:02d}-gate.md`: (1) Clarifying Assumptions, (2) Pattern/Structure Chosen & Why, (3) Time & Space Complexity Bound, (4) Failure Mode Under 10x Load or Malicious Input, and (5) Primary Design Trade-Off Accepted.",
            f"- [ ] **Task 5: Clean-Room Verification & Failure Diagnosis (7m):** Execute clean-room rebuild from scratch, verify zero warnings in `pytest`, `mypy --strict`, and `ruff check`, and record failure diagnosis notes in `artifacts/day{day_num:03d}/failure-notes.md`."
        ]

    # Category C: Mode B Adversarial AI Verification Labs (Day 6 of Weeks 5-32)
    if day_num in MODE_B_AI_AUDITS:
        audit_focus, flaw_proof = MODE_B_AI_AUDITS[day_num]
        if day_num in PREDICT_AND_TRACE_DAYS:
            trap_desc = PREDICT_AND_TRACE_TRAPS.get(day_num, trap_warning)
            return [
                f"- [ ] **[CORE] Task 1: Setup, Reading & Predict & Trace Checkpoint (12m + 3m):** Open `py-mastery/`, check `git status`, create `artifacts/day{day_num:03d}/`, and review the mapped book section. Read the 10–15 line trap snippet embodying today's Engineering Trap to Avoid ({trap_desc}), predict the exact runtime output or exception in `artifacts/day{day_num:03d}/prediction.txt` before executing it, run it to verify, and explain any discrepancy.",
                f"- [ ] **[CORE] Task 2: From-Scratch Implementation (20m):** {practical} (AI autocomplete strictly OFF).",
                f"- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` unit and edge-case tests verifying `{title}` and save the test log beside {deliverable}.",
                f"- [ ] **[MINOR] Task 4: Mode B Constrained AI Audit (10m):** {audit_focus}",
                f"- [ ] **Task 5: Mode B Hallucination Proof & Artifact Logging (8m):** {flaw_proof} Record prompt, caught hallucination, and empirical proof in `artifacts/day{day_num:03d}/ai-audit.md`."
            ]
        else:
            return [
                f"- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day{day_num:03d}/`, and review the mapped book section for `{title}`.",
                f"- [ ] **[CORE] Task 2: From-Scratch Implementation (25m):** {practical} (AI autocomplete strictly OFF).",
                f"- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` unit and edge-case tests verifying `{title}` and save the test log beside {deliverable}.",
                f"- [ ] **[MINOR] Task 4: Mode B Constrained AI Audit (10m):** {audit_focus}",
                f"- [ ] **Task 5: Mode B Hallucination Proof & Artifact Logging (8m):** {flaw_proof} Record prompt, caught hallucination, and empirical proof in `artifacts/day{day_num:03d}/ai-audit.md`."
            ]

    # Category D: Predict & Trace Days (Days 2-5, 8, 15-18, etc.)
    if day_num in PREDICT_AND_TRACE_DAYS:
        trap_desc = PREDICT_AND_TRACE_TRAPS.get(day_num, trap_warning)
        task5 = ""
        if day_num % 2 == 0:
            bug_desc = MICRO_BUG_DRILLS.get(day_num, "subtle invariant or boundary defect")
            task5 = f"- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the failing symptom or subtle defect in `drills/day{day_num:03d}_buggy.py` ({bug_desc}). Formulate a 1-sentence falsifiable root-cause hypothesis in `artifacts/day{day_num:03d}/failure-notes.md` before editing code, then fix the defect and verify the regression test suite passes within 7 minutes."
        else:
            task5 = f"- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately break one invariant or type/boundary assumption in `{title}`, verify that `pytest` or `mypy --strict` catches it immediately, restore the fix, and log the symptom in `artifacts/day{day_num:03d}/failure-notes.md`."

        return [
            f"- [ ] **[CORE] Task 1: Setup, Reading & Predict & Trace Checkpoint (12m + 3m):** Open `py-mastery/`, check `git status`, create `artifacts/day{day_num:03d}/`, and review the mapped book section. Read the 10–15 line trap snippet embodying today's Engineering Trap to Avoid ({trap_desc}), predict the exact runtime output or exception in `artifacts/day{day_num:03d}/prediction.txt` before executing it, run it to verify, and explain any discrepancy.",
            f"- [ ] **[CORE] Task 2: From-Scratch Implementation (20m):** {practical} (AI autocomplete strictly OFF).",
            f"- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` unit and edge-case tests verifying `{title}` and save the test log beside {deliverable}.",
            f"- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** {minor_focus}: {minor_cmd}",
            task5
        ]

    # Category E: Standard Days
    task5 = ""
    if day_num % 2 == 0:
        bug_desc = MICRO_BUG_DRILLS.get(day_num, "subtle invariant or boundary defect")
        task5 = f"- [ ] **Task 5: Pre-Seeded Micro-Bug Diagnosis (7m):** Inspect the failing symptom or subtle defect in `drills/day{day_num:03d}_buggy.py` ({bug_desc}). Formulate a 1-sentence falsifiable root-cause hypothesis in `artifacts/day{day_num:03d}/failure-notes.md` before editing code, then fix the defect and verify the regression test suite passes within 7 minutes."
    else:
        task5 = f"- [ ] **Task 5: Failure & Diagnosis Test (7m):** Deliberately break one invariant or type/boundary assumption in `{title}`, verify that `pytest` or `mypy --strict` catches it immediately, restore the fix, and log the symptom in `artifacts/day{day_num:03d}/failure-notes.md`."

    return [
        f"- [ ] **[CORE] Task 1: Setup & Reading (10m):** Open `py-mastery/`, check `git status`, create `artifacts/day{day_num:03d}/`, and review the mapped book section for `{title}`.",
        f"- [ ] **[CORE] Task 2: From-Scratch Implementation (25m):** {practical} (AI autocomplete strictly OFF).",
        f"- [ ] **[CORE] Task 3: Automated Unit Tests (7m):** Write `pytest` unit and edge-case tests verifying `{title}` and save the test log beside {deliverable}.",
        f"- [ ] **[MINOR] Task 4: Quality & Tooling Audit (11m):** {minor_focus}: {minor_cmd}",
        task5
    ]

DAY_METADATA_OVERRIDES = {
    1: {
        "objective": "Initialize modern src/ layout with pyproject.toml (PEP 621), configure ruff, mypy --strict, and pytest, and master diagnostic triage of static invariance errors and runtime test failure diffs."
    },
    19: {
        "objective": "Master advanced pytest idioms (fixtures, tmp_path, monkeypatch, caplog, parametrization) and audit AI-generated test suites to eliminate tautological assertions."
    },
    20: {
        "objective": "Author stateful test doubles (FakeClock, FakeMetricsSink) from scratch and audit AI test suites to replace brittle unittest.mock.patch calls with reliable fakes."
    },
    21: {
        "title": "Week 3 Gate: Fault-Tolerant Resource Manager, Test Audit & Git Bisect Archaeology",
        "objective": "Consolidate Week 3 packaging, context managers, and test doubles, execute git bisect run pytest across a 10-commit history branch to isolate a resource leak, and deliver oral Hiring Committee Packet Narration."
    },
    28: {
        "title": "Week 4 Gate (Month 1 Review): Systems Architecture Defense & PR Diff Code Review",
        "objective": "Consolidate Month 1 core systems Python from memory, execute a structured PR diff code review critique on 150 lines of AI-generated churn, and deliver oral Hiring Committee Packet Narration."
    },
    35: {
        "title": "Week 5 Gate: Unfamiliar Codebase Comprehension (Sequence Cache) & Slotted Keys",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week05_repo/: cold trace call graph, squash 2 seeded bugs, implement dynamic capacity resizing, and record Hiring Committee Packet."
    },
    44: {
        "objective": "Implement FairRequestScheduler using heapq with custom priority tie-breaking, and solve live interview constraint shifts by implementing monotonic credit aging to prevent starvation (Anthropic scheduler benchmark)."
    },
    46: {
        "objective": "Implement an O(1) LRUCache from scratch using Doubly Linked List + Dict, and solve live interview constraint shifts by adding byte-size weighting (max_bytes), monotonic TTL expiration, and randomized reference-oracle fuzzing."
    },
    47: {
        "objective": "Implement TokenPrefixTrie for KV-cache prefix matching, and solve live interview constraint shifts by adding reference-counted prefix block pinning (ref_count > 0) preventing eviction during active generation."
    },
    49: {
        "title": "Week 7 Gate: Unfamiliar Codebase Comprehension (Scheduler & Trie) & Benchmark",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week07_repo/: cold trace call graph, squash 2 seeded bugs, implement starvation-free credit aging, and record Hiring Committee Packet."
    },
    51: {
        "objective": "Implement atomic file writes with pathlib and temporary files, hardened with path-traversal (../) and symlink-escape sanitization (Path.resolve() containment check)."
    },
    52: {
        "objective": "Parse SafeTensors binary headers from scratch with struct, bridge conceptual zero-copy slicing with mmap/memoryview, and enforce adversarial validation rejecting 64-bit header-length bombs (> 100 MB)."
    },
    53: {
        "objective": "Implement memory-mapped zero-copy binary tensor slicing via mmap and memoryview, adding adversarial validation rejecting overlapping or out-of-bounds tensor data offsets."
    },
    56: {
        "title": "Week 8 Gate (Month 2 Review): 4-Level Progressive Refactoring Simulation & PR Diff Critique",
        "objective": "Execute an Anthropic OA-style 4-Level Progressive Refactoring Simulation on an in-memory caching engine: unlock Level 1 (core container) -> Level 2 (filtering/sorting) -> Level 3 (monotonic TTL & rollback) -> Level 4 (batch compaction & serialization), critique a 150-line AI PR diff, and record Hiring Committee Packet."
    },
    63: {
        "title": "Week 9 Gate: Unfamiliar Codebase Comprehension (Async Task Supervisor)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week09_repo/: cold trace async execution graph, squash 2 seeded concurrency bugs, implement task heartbeat supervisor, and record Hiring Committee Packet."
    },
    68: {
        "objective": "Implement AsyncDynamicBatcher with per-request Futures, and solve live interview constraint shifts by handling caller timeout cancellation after queuing without raising InvalidStateError."
    },
    69: {
        "objective": "Implement TokenBucketRateLimiter with monotonic clock refill, thread/async safety, and per-tenant queueing, auditing AI implementations for wall-clock drift and missing locks (Google May 2026 Code Comprehension Benchmark)."
    },
    73: {
        "objective": "Parse HTTP/1.1 wire chunks and Server-Sent Events (text/event-stream) with zero external dependencies, hardened with a strict max_line_bytes buffer cap preventing unbounded memory-exhaustion DoS."
    },
    75: {
        "objective": "Implement raw ASGI specification from scratch without web frameworks, hardened with request body byte caps (max_body_bytes) and header read timeouts defending against Slowloris attacks."
    },
    77: {
        "title": "Week 11 Gate: Unfamiliar Codebase Comprehension (Raw ASGI Streaming Server)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week11_repo/: cold trace ASGI scope lifecycle, squash 2 seeded streaming bugs, implement request body chunking, and record Hiring Committee Packet."
    },
    84: {
        "title": "Week 12 Gate (Month 3 Review): Multi-Process Async Streaming Service & Git Bisect",
        "objective": "Consolidate multi-process async streaming service, execute git bisect run pytest across a 10-commit history branch to isolate latency regressions, and deliver oral Hiring Committee Packet Narration."
    },
    89: {
        "objective": "Refactor legacy/messy Python into clean, typed, modular code, and perform a structured PR diff code review critique on 150 lines of AI-generated code churn (GitClear pattern) in artifacts/day089/pr-review.md."
    },
    91: {
        "title": "Week 13 Gate: Unfamiliar Codebase Comprehension (Plugin Registry & DI Container)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week13_repo/: cold trace plugin registration, squash 2 seeded lifecycle bugs, implement scoped resolution, and record Hiring Committee Packet."
    },
    94: {
        "objective": "Implement TOML configuration parsing with tomllib, environment variable overrides, SecretStr masking, and constant-time token comparison (hmac.compare_digest) preventing timing attacks."
    },
    109: {
        "title": "Closed-Book 4-Level Progressive Refactoring Simulation (Anthropic OA Timed Challenge)",
        "objective": "Execute a timed 60-minute Anthropic 4-Level Progressive Refactoring Challenge: unlock Level 1 (base container) -> Level 2 (prefix filtering) -> Level 3 (monotonic TTL & rollback) -> Level 4 (batch compaction & serialization) under strict time pressure."
    },
    119: {
        "title": "Week 17 Gate: Unfamiliar Codebase Comprehension (MiniModule Parameter Tree)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week17_repo/: cold trace parameter registration, squash 2 seeded recursion bugs, implement named parameter filtering, and record Hiring Committee Packet."
    },
    120: {
        "objective": "Master runtime inspection with inspect.signature and wire the foundational architecture for an Agentic Tool-Call Execution Sandbox that introspects function parameters and emits strict JSON Schema."
    },
    133: {
        "title": "Week 19 Gate: Unfamiliar Codebase Comprehension (Dynamic Route Dispatcher)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week19_repo/: cold trace route matching, squash 2 seeded regex/exception bugs, implement sliding-window rate limiting, and record Hiring Committee Packet."
    },
    136: {
        "objective": "Implement NanoServe route registration and HTTP handler, hardened with strict payload size limits and URL validation defending against SSRF attacks on loopback and metadata addresses."
    },
    147: {
        "title": "Week 21 Gate: Unfamiliar Codebase Comprehension (Zero-Copy Paged Arena)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week21_repo/: cold trace slab allocation, squash 2 seeded buffer bugs, implement slab compaction, and record Hiring Committee Packet."
    },
    161: {
        "title": "Week 23 Gate: Unfamiliar Codebase Comprehension (Pure-Python Transformer)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week23_repo/: cold trace attention execution graph, squash 2 seeded RoPE/KV bugs, implement temperature sampling, and record Hiring Committee Packet."
    },
    175: {
        "title": "Week 25 Gate: Unfamiliar Codebase Comprehension (Stack-Sampling Profiler)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week25_repo/: cold trace profiler sampling loop, squash 2 seeded frame/time bugs, implement allocation hotspot analysis, and record Hiring Committee Packet."
    },
    180: {
        "objective": "Audit provided C99 reference template fast_ops.c containing 2 seeded pointer-lifetime and alignment bugs, fix defects, declare strict ctypes bindings, pin memoryview lifetimes against premature GC, and release the GIL."
    },
    184: {
        "objective": "Audit provided C++ reference template paged_attention_binding.cpp containing 2 seeded pointer-lifetime and buffer bugs, fix defects, declare strict nanobind/pybind11 exports, pin GC lifetimes, and release the GIL."
    },
    189: {
        "title": "Week 27 Gate: Unfamiliar Codebase Comprehension (Native C/CUDA FFI Bridge)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week27_repo/: cold trace FFI boundaries, squash 2 seeded ctypes/GC bugs, implement DLPack capsule export, and record Hiring Committee Packet."
    },
    203: {
        "title": "Week 29 Gate: Unfamiliar Codebase Comprehension (Distributed TP Worker Group)",
        "objective": "Execute a 60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation on comprehension_labs/week29_repo/: cold trace worker IPC protocol, squash 2 seeded reduction/barrier bugs, implement prefix-aware routing, and record Hiring Committee Packet."
    },
    204: {
        "objective": "Clone official open-source vllm repository, trace vllm/v1/engine/async_llm.py, vllm/v1/engine/core.py, and vllm/v1/core/sched/scheduler.py with Mode B constrained prompts, and implement clean-room AsyncLLM engine separation in src/pymastery/vllm_mini/engine.py."
    },
    205: {
        "objective": "Trace vllm/transformers_utils/detokenizer.py with Mode B constrained prompts, and implement clean-room IncrementalDetokenizer in src/pymastery/vllm_mini/detokenizer.py with UTF-8 subword holdback."
    },
    206: {
        "objective": "Trace vllm/entrypoints/openai/serving_chat.py and protocol.py with Mode B constrained prompts, and implement clean-room OpenAI /v1/chat/completions protocol in src/pymastery/vllm_mini/protocol.py."
    },
    207: {
        "objective": "Trace vllm/model_executor/guided_decoding/ with Mode B constrained prompts, implement guided JSON logits masking, and connect Day 120 signature introspection to wire a deterministic Agentic Tool-Call Execution Sandbox & Evaluation Harness."
    },
    208: {
        "objective": "Trace vllm/lora/worker_manager.py and models.py with Mode B constrained prompts, and implement clean-room LoRACacheManager in src/pymastery/vllm_mini/lora.py with LRU eviction and active slot pinning."
    },
    209: {
        "objective": "Trace vllm/engine/metrics.py with Mode B constrained prompts, implement clean-room EngineObservabilityHook in src/pymastery/vllm_mini/metrics.py, and audit AI-generated Prometheus metrics for wall-clock drift (Google May 2026 Benchmark)."
    }
}
