from pathlib import Path
import html as html_lib
import re
import shutil
import subprocess

from python_curriculum_part1 import WEEKS_PART1
from python_curriculum_part2 import WEEKS_PART2
import curriculum_enhancements

ROOT = Path(__file__).parent
md_path = ROOT / "PYTHON_MASTERY_CURRICULUM.md"
html_path = ROOT / "PYTHON_MASTERY_CURRICULUM_BOOK.html"
pdf_path = ROOT / "PYTHON_MASTERY_CURRICULUM_BOOK.pdf"
ref_html_path = ROOT / "AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html"

HEADER_MD = """# Python Mastery for AI Inference Engineering: 8-Month From-Scratch Curriculum

## 1. Curriculum Analysis & Language Selection

### 1.1 Findings from `AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html`

An exhaustive audit of your [AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html](file:///D:/AI%20Inference%20Engineering/Resources/AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html) (and its source [AI_INFERENCE_ENGINEERING_CURRICULUM.md](file:///D:/AI%20Inference%20Engineering/Resources/AI_INFERENCE_ENGINEERING_CURRICULUM.md)) reveals a decisive architectural reality across both **Path A (AI Inference Systems / ML Infrastructure / Model Serving Engineering)** and **Path B (Advanced AI Systems / GPU / Low-Level Inference Engineering)**:

| Language Candidate | Role in `AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html` | Track 1 (Weeks 1–16) Share | Track 2 (Modules 2A–2G) Role | Verdict as Starting Language to Master |
|---|---|---|---|---|
| **Python (3.12+)** | **Primary Control-Plane, Tooling, Serving, Benchmarking, Tensor & DSL Language** (`MOVE EARLIER / KEEP` to Day 1) | **~85% of all daily code** (Typed clients, `asyncio`, SSE streaming, PyTorch tensors/attention, KV-cache math, vLLM/SGLang orchestration, FastAPI gateways, Prometheus metrics, chaos harnesses) | **Primary host & DSL language** for Triton GPU kernels (`@triton.jit`, Module 2D), vLLM scheduler/block-manager internals (Module 2E), multi-GPU orchestration (Module 2F), and paper reproduction (Module 2G) | **#1 MOST PREFERRED STARTING LANGUAGE** |
| **C / Modern C++** | Low-level runtime & kernel binding language (`DEFER` / `REDUCE` in Section 6 Change Log) | **0% required in Track 1** (explicitly deferred so it does not block career entry) | Modules 2A, 2B, 2C (manual memory, RAII, CUDA kernels wrapped into Python via `pybind11`/`nanobind`) | **#2 Language (Master after/via Month 7 Python C/C++ FFI bridge)** |
| **CUDA C / PTX** | Hardware-specific GPU kernel language | Profiling awareness only (`nsys` CLI in Week 7) | Module 2C (custom GPU kernel authoring) | **#3 Specialized Kernel Language** |
| **Bash / Linux Shell** | Operational inspection glue (`ps`, `/proc`, `docker`, `nvidia-ctk`) | Supporting 30% minor focus | Supporting tooling | **Supporting Operational Tool, not primary programming language** |

> [!IMPORTANT]
> **Why Python 3.12+ is the Unquestioned #1 Language to Master First:**
> 1. **100% of Modern Inference Control Planes are Written in Python:** vLLM, SGLang, TensorRT-LLM's frontend, PyTorch (`torch.compile` / TorchDynamo), Hugging Face Transformers, and OpenAI-compatible gateways are architected in Python. Specifically, vLLM's Continuous Batching Scheduler, PagedAttention Block Manager, Prefix Caching tree, AsyncLLMEngine, ZeroMQ/SharedMemory IPC loop, and API server are all pure Systems Python.
> 2. **Triton GPU Kernels are Written in Python:** In Track 2 (Module 2D), high-performance fused GPU kernels (like FlashAttention, fused RMSNorm, and quantized matmuls) are written using OpenAI's **Triton** language—a Python AST/decorator DSL (`@triton.jit`).
> 3. **Python is the Bridge to C, C++, and CUDA:** By mastering Python deeply (up through the C-level Buffer Protocol, `memoryview`, `ctypes`, `pybind11`/`nanobind`, and DLPack in Months 6–7 of this curriculum), you automatically learn how memory layouts, pointers, strides, and GIL-free native C/C++/CUDA extensions work.

---

### 1.2 The "Quality Code From Scratch" Philosophy (No Tutorial Copy-Pasting, No Basic Syntax Fluff)

Because you **already know basic programming concepts** (variables, loops, conditionals, basic functions, lists/dicts), this curriculum **skips elementary syntax** on Day 1. Instead, you will spend 8 months writing **production-grade, strictly-typed, tested, high-performance Python code completely by hand**:

1. **Build Before You Import:** Before using external frameworks like Pydantic, FastAPI, NumPy, PyTorch, Prometheus, or vLLM, you will **implement clean working versions of their core mechanisms yourself from scratch** using the Python standard library (`NanoSchema`, `NanoServe`, `StridedTensor`, `AutoTensor`, `TokenPrefixTrie`, `PagedKVBlockAllocator`, `ContinuousBatchScheduler`, `SamplingFlamegraphProfiler`, `MetricsRegistry`, and `Mini-vLLM`).
2. **Zero-Compromise Engineering Standards on Every File:**
   - **100% Static Type Annotations:** Every function, method, generator, and class attribute must pass `mypy --strict` (no untyped defs, no implicit `Any`, no `# type: ignore` shortcuts).
   - **Zero Linter or Formatting Warnings:** Every file passes `ruff check` and `ruff format`.
   - **Deterministic Resource & Memory Hygiene:** Every file, socket, subprocess, `SharedMemory` segment, and async task is managed via context managers (`with` / `async with` / `ExitStack`) and verified for zero leaks using `tracemalloc`.
   - **Self-Written Automated Tests:** Every single day includes a `pytest` verification task covering normal behavior, boundary conditions, and a deliberate failure/chaos test.

---

### 1.3 The 8-Month (224-Day, 1 Hour/Day) Architecture & Daily Time Contract

- **Total Duration:** **8 Months** = **32 Weeks** = **224 Daily Lessons** (7 focused study/build days per week, **1.0 Hour per Day** = **224 Total Hours**).
  - **Phase 1 — Base Concepts Mastery (Months 1–4 | Weeks 1–16 | Days 1–112 | 112 Hours):**
    Transforms basic Python familiarity into rock-solid **Production Core & Intermediate Systems Python**: modern packaging (`pyproject.toml`), strict static typing (`Protocol`, `Generic`, `TypeGuard`), frozen/slotted dataclasses, defensive context management (`ExitStack`), lazy O(1)-memory generator pipelines, the Python Data Model (dunder protocols, `__slots__`, `weakref`), closures/decorators/descriptors, algorithmic data structures (`deque`, `heapq`, `bisect`, O(1) LRU Cache, Prefix Radix Trie), binary/mmap SafeTensors I/O, deep `asyncio` structured concurrency (`TaskGroup`, cancellation, dynamic batching, raw TCP/SSE/ASGI streaming), `multiprocessing` + `shared_memory`, and the **Month 4 Base Capstone (`pyserve-core`)**.
  - **Phase 2 — Advanced Concepts Mastery (Months 5–8 | Weeks 17–32 | Days 113–224 | 112 Hours):**
    Elevates you to **Senior/Staff Systems & AI Inference Python**: runtime metaprogramming (`__init_subclass__`, metaclasses, AST transformations, `dis` bytecode inspection, `sys.monitoring`), authoring a Pydantic + FastAPI framework from scratch (`NanoSchema` + `NanoServe`), CPython memory internals (`PyObject`, refcounting, cyclic GC tuning, Buffer Protocol, zero-copy `memoryview`), building a strided N-dimensional Tensor, Quantizer (INT8/INT4), Autograd engine, and KV-cached Transformer (`NanoLLM`) from scratch, implementing vLLM's PagedAttention Block Manager & Continuous Batching Scheduler, deterministic/statistical flamegraph profiling, roofline analysis, Free-Threaded Python (PEP 703), native C/C++/CUDA/Triton/DLPack interoperability, distributed multi-process IPC, and the **Month 8 Advanced Capstone (`Mini-vLLM`)**.

#### The 60-Minute Daily Workload Contract: Dual-Mode Weekly Rhythm

Every daily lesson is strictly bounded to **1.0 Hour (60 minutes)** so you can sustain a 100% streak alongside your other commitments: **Full Session: 60 mins | Core-Only Sprint: 42 mins | Minor/Verification: 18 mins**. The curriculum operates on an interview-aligned **Dual-Mode Weekly Rhythm**:

##### 1. Mode A: Blank-File Construction & Micro-Diagnosis (Days 1–5 of Each Week)
- **Block 1 (15 mins | Core Reading + Predict & Trace):** 12 mins targeted book/doc retrieval + 3 mins pre-execution "Predict & Trace" on a 10–15 line snippet demonstrating that day's yellow Engineering Trap to Avoid.
- **Block 2 (30 mins | [CORE - 70%] From-Scratch Implementation):** Blank-file implementation in `src/pymastery/` with AI autocomplete strictly OFF. You type every class, type signature, and test by hand to build indestructible muscle memory.
- **Block 3 (15 mins | [MINOR - 30%] Verification, Static Audit & Diagnostic Task 5):** 8 mins `pytest` + `mypy --strict` + `ruff check`, followed by 7 mins Task 5 (Odd days: self-inflicted invariant break to test unit-test sensitivity; Even days: pre-seeded micro-bug diagnosis in `drills/dayXXX_buggy.py`).

##### 2. Mode B: Adversarial AI Verification Lab (Day 6 of Each Week, Month 2 Onward)
- Dedicate the final 20 minutes of the session to practicing the human-led, AI-assisted engineering workflow graded in Google and Meta coding interviews. Using constrained, hypothesis-driven prompts, inspect an unfamiliar module or audit an AI-generated implementation containing subtle concurrency, memory, or security flaws, logging empirical failure proofs in `artifacts/dayXXX/ai-audit.md`.

##### 3. Weekly Gate: Unfamiliar Codebase Comprehension & Packet Narration (Day 7 of Each Week)
- Combines clean-room verification, 3-phase unfamiliar codebase comprehension/extension (on 12 dedicated benchmark weeks modeled on Stripe Bug Squash / Google Code Comprehension rounds), and a mandatory 3-minute oral Hiring Committee Packet Narration covering assumptions, architecture, Big-O bounds, 10x failure modes, and trade-offs in `gates/weekXX-gate.md`.

| Weekly Day | Mode / Focus | Core Blocks (42 mins) | Minor / Audit Blocks (18 mins) | Primary Verification Deliverable |
|---|---|---|---|---|
| **Days 1–5** | **Mode A (Blank-File Build & Diagnosis)** | 12m Reading + 3m Predict & Trace (Block 1) + 27m Blank-File Build (Block 2) | 11m Static Audit (`mypy`/`ruff`) + 7m Task 5 (Micro-Bug or Invariant Break) | `src/pymastery/` module + `drills/` or `artifacts/` failure proof |
| **Day 6 (W5–32)** | **Mode B (Adversarial AI Verification)** | 10m Reading + 25m Hand-Typed Core + 7m Unit Tests | 10m Constrained AI Audit + 8m Hallucination Proof (`pytest`/`tracemalloc`) | `artifacts/dayXXX/ai-audit.md` empirical test failure proof |
| **Day 7 (All Weeks)** | **Weekly Gate & Packet Narration** | 10m Cold Trace / Baseline + 22m Implementation/Extension + 10m Test Suite | 10m Static/Bottleneck Audit + 8m Hiring Committee Packet Narration Check | `gates/weekXX-gate.md` 5-bullet summary + passing gate suite |

---

## 2. Comprehensive Topic Mastery Matrix (Months 1–8)

| Month | Phase | Weeks (Days) | Core Engineering Theme | Major From-Scratch Implementations |
|---|---|---|---|---|
| **Month 1** | **Base (Part 1)** | Weeks 1–4 (Days 1–28) | **Modern Packaging, Strict Typing, Data Modeling, Context Safety & Lazy Generators** | `pyproject.toml` layout, `Result[T, E]` union, `RingBuffer[T]`, `ExitStack` transactional writer, O(1)-memory streaming JSONL benchmark log analyzer CLI |
| **Month 2** | **Base (Part 2)** | Weeks 5–8 (Days 29–56) | **Python Data Model, `__slots__`, Descriptors, Algorithmic Collections & Binary/Mmap I/O** | Slotted cache keys, `PureProperty` descriptor, `@retry` typed decorator, `FairRequestScheduler` (`heapq`), O(1) `LRUCache`, `TokenPrefixTrie`, `mmap` `.safetensors` binary parser |
| **Month 3** | **Base (Part 3)** | Weeks 9–12 (Days 57–84) | **Deep `asyncio` Concurrency, Dynamic Batching, Raw TCP/SSE/ASGI & Multiprocessing IPC** | `TaskGroup` supervisor, `AsyncDynamicBatcher`, length-prefixed TCP RPC, zero-dependency `SSEParser`, raw ASGI streaming server with graceful drain, `SharedMemory` zero-copy IPC |
| **Month 4** | **Base (Part 4)** | Weeks 13–16 (Days 85–112) | **Pythonic Design Patterns, Property Testing, SQLite Persistence & Base Mastery Capstone** | Strategy/Registry samplers, property-based state fuzzer, SQLite WAL benchmark repo, and **`pyserve-core` (Month 4 Base Mastery Capstone)** |
| **Month 5** | **Advanced (Part 1)** | Weeks 17–20 (Days 113–140) | **Metaprogramming, Metaclasses, AST/Bytecode Compilation, DI & Framework Authoring** | `MiniModule` (`nn.Module` clone), AST linter & constant folder, bytecode inspector, symbolic graph tracer, **`NanoSchema` (Pydantic clone)** & **`NanoServe` (FastAPI clone)** |
| **Month 6** | **Advanced (Part 2)** | Weeks 21–24 (Days 141–168) | **CPython Memory Internals, Buffer Protocol, Strided Tensors, Autograd, LLM & Paged KV Cache** | `SlabArenaAllocator` (`memoryview`), `StridedTensor` (zero-copy views & broadcasting), INT8/INT4 quantizer, `AutoTensor` autograd, `NanoLLM` (RoPE, GQA, KV Cache, BPE), **PagedAttention Block Manager & Continuous Batching Scheduler** |
| **Month 7** | **Advanced (Part 3)** | Weeks 25–28 (Days 169–196) | **Profiling, Flamegraphs, Roofline, GIL/Free-Threading & Native C/C++/CUDA/Triton Interop** | `SamplingFlamegraphProfiler`, `AllocationHotspotAnalyzer`, `RooflineModel`, zero-copy `ctypes` C kernels (`fast_ops.c`), DLPack bridge, `MetaTensor` 0-byte verifier, Tiled FlashAttention & CUDA Graph simulator |
| **Month 8** | **Advanced (Part 4)** | Weeks 29–32 (Days 197–224) | **Distributed IPC, Tensor Parallelism, Prometheus Engine, vLLM Architecture & Final Capstone** | Multi-rank TP worker group, Prometheus exposition engine, `PrefixAwareLoadBalancer`, guided JSON logits masker, LoRA cache manager, and **`Mini-vLLM` (Month 8 Advanced Mastery Capstone)** |
"""

def build_days_markdown(weeks_list, phase_title, phase_intro):
    out = [f"\n---\n\n## {phase_title}\n\n{phase_intro}\n"]
    day_counter = 1 if weeks_list[0][0] == 1 else 113
    current_month = None
    month_titles = {
        1: "Month 1: Modern Packaging, Strict Typing, Error Safety & Lazy Generators (Weeks 1–4, Days 1–28)",
        2: "Month 2: Python Data Model, Descriptors, Algorithmic Structures & Binary I/O (Weeks 5–8, Days 29–56)",
        3: "Month 3: Deep Asyncio, Streaming Protocols, ASGI & Multiprocessing Systems (Weeks 9–12, Days 57–84)",
        4: "Month 4: Pythonic Architecture, SQLite Storage & Base Mastery Capstone (Weeks 13–16, Days 85–112)",
        5: "Month 5: Metaprogramming, AST/Bytecode & Building FastAPI/Pydantic From Scratch (Weeks 17–20, Days 113–140)",
        6: "Month 6: CPython Memory, Zero-Copy Buffers, Strided Tensors, LLM & Paged KV Cache (Weeks 21–24, Days 141–168)",
        7: "Month 7: Profiling, Roofline, Free-Threaded Python & Native C/C++/GPU Interop (Weeks 25–28, Days 169–196)",
        8: "Month 8: Distributed IPC, Observability, vLLM Internals & Advanced Capstone (Weeks 29–32, Days 197–224)",
    }
    for week_num, month_num, week_title, big_picture, minor_focus, minor_why, mapped_resource, minor_cmd_template, trap_warning, days in weeks_list:
        if month_num != current_month:
            current_month = month_num
            out.append(f"\n### {month_titles[month_num]}\n")
        out.append(f"\n#### Week {week_num} - {week_title}\n")
        track_badge = "Base Track (Months 1-4)" if month_num <= 4 else "Advanced Track (Months 5-8)"
        for title, objective, practical, deliverable, stop_cond, dont_study in days:
            minor_cmd = minor_cmd_template.format(day=day_counter)

            # Check for day metadata overrides
            if day_counter in curriculum_enhancements.DAY_METADATA_OVERRIDES:
                overrides = curriculum_enhancements.DAY_METADATA_OVERRIDES[day_counter]
                title = overrides.get("title", title)
                objective = overrides.get("objective", objective)
                practical = overrides.get("practical", practical)
                deliverable = overrides.get("deliverable", deliverable)
                stop_cond = overrides.get("stop_cond", stop_cond)
                dont_study = overrides.get("dont_study", dont_study)

            # Determine Big Picture text
            if day_counter in curriculum_enhancements.COMPREHENSION_GATES:
                big_pic = (
                    f"Today is an intensive **60-Minute 3-Phase Unfamiliar Codebase Comprehension Simulation** "
                    f"modeled on the Google Code Comprehension Round, Meta AI Coding Interview, and Stripe Bug Squash. "
                    f"You will cold-read an unfamiliar 300–500 line multi-file repository in `comprehension_labs/week{week_num:02d}_repo/`, "
                    f"squash 2 seeded bugs, implement an architectural feature extension, profile bottlenecks, and narrate your Hiring Committee Packet."
                )
            elif day_counter in [56, 109]:
                big_pic = (
                    f"{big_picture}\n>\n> Today is an Anthropic OA-style **4-Level Progressive Refactoring Simulation**: "
                    f"Level k+1 requirements unlock only after Level k unit tests pass. You will scale from a core in-memory container "
                    f"to prefix/range filtering, monotonic TTL rollback, and thread-safe batch compaction within the 60-minute contract without ground-up rewrites."
                )
            elif day_counter in range(204, 210):
                big_pic = (
                    f"{big_picture}\n>\n> **Grounded in Actual Open-Source vLLM Source Files:** Clone the official `vllm` repository alongside `py-mastery/` "
                    f"and trace production source files with Mode B constrained prompts before implementing the clean-room `src/pymastery/vllm_mini/` module."
                )
            elif day_counter in curriculum_enhancements.MODE_B_AI_AUDITS:
                big_pic = (
                    f"{big_picture}\n>\n> Today's session includes a **Mode B Adversarial AI Verification Lab** (final 18 mins): "
                    f"after hand-typing your from-scratch implementation with autocomplete OFF, you will use constrained prompts to audit an AI-generated "
                    f"implementation, catch subtle hallucinations, and log empirical failure proofs in `artifacts/day{day_counter:03d}/ai-audit.md`."
                )
            elif day_counter % 7 == 0 and day_counter not in curriculum_enhancements.COMPREHENSION_GATES:
                big_pic = (
                    f"{big_picture}\n>\n> Today is your **Weekly Gate & Hiring Committee Defense**: consolidate weekly systems invariants cleanly from scratch, "
                    f"complete clean-room verification, and record your 5-bullet oral Hiring Committee Packet Summary in `gates/week{week_num:02d}-gate.md`."
                )
            else:
                big_pic = (
                    f"{big_picture}\n>\n> Today is a focused **1.0-hour engineering session**: 42m core systems focus (`[CORE - 70%]`) "
                    f"and 18m static verification, quality audit, and diagnostic/AI audit (`[MINOR - 30%]`)."
                )

            tasks = curriculum_enhancements.generate_day_tasks(
                day_counter,
                week_num,
                month_num,
                title,
                objective,
                practical,
                deliverable,
                stop_cond,
                dont_study,
                minor_focus,
                minor_cmd,
                trap_warning,
            )
            tasks_str = "\n".join(tasks)

            block = f"""##### Day {day_counter}: {title} `[CORE: {track_badge}]` `[MINOR: {minor_focus}]`

> [!NOTE]
> **The Big Picture:** {big_pic}

- **Primary Focus [CORE - 70% Weight | 42 mins]:** {objective}
- **Secondary Focus [MINOR - 30% Weight | 18 mins]:** {minor_focus} — because {minor_why}.
- **Mapped Book & Docs:** {mapped_resource}

###### Step-by-Step To-Do Tasks:
{tasks_str}

> [!WARNING]
> **Engineering Trap to Avoid:** {trap_warning}

> [!IMPORTANT]
> **Daily Gate (STOP WHEN):** {stop_cond}

- **Tangible Deliverable:** {deliverable}
- **STOP WHEN:** {stop_cond}
- **Do Not Study Yet:** {dont_study}
- **Estimated Time:** 1.0 Hour total (Full Session: 60 mins | Core-Only Sprint: 42 mins)

"""
            out.append(block)
            day_counter += 1
    return "".join(out)

FOOTER_MD = """
---

## 5. Topic Boundary Cards (Months 1–8)

### Card 1: Modern Packaging, Strict Typing & Data Modeling (Month 1)
- **Engineering Purpose:** Eliminate whole classes of `NoneType`, schema mismatch, and environment drift bugs before runtime.
- **Required Depth:** Level 4 (From-Scratch Implementation & Strict Enforcement).
- **Must Implement Yourself:** `pyproject.toml` (PEP 621), `dataclass(frozen=True, slots=True)`, `Enum`/`StrEnum`, `Protocol`, `Generic[T]`, `ParamSpec`, `TypeGuard`, and `Result[T, E]`.
- **STOP WHEN:** Every module in `src/pymastery/` passes `mypy --strict` and `ruff check` with zero warnings and 0 uses of `Any`.

### Card 2: Context Safety, Testing Craftsmanship & Lazy Generators (Month 1)
- **Engineering Purpose:** Guarantee deterministic resource cleanup under exceptions and O(1) memory footprint on multi-gigabyte data streams.
- **Required Depth:** Level 4 (Implementation + Memory Verification).
- **Must Implement Yourself:** Custom exception trees with `__cause__`, `ExceptionGroup`/`except*`, `__enter__`/`__exit__`, `ExitStack`, custom `Iterable`/`Iterator` classes, and lazy generator pipelines verified with `tracemalloc`.
- **STOP WHEN:** Streaming 250,000 JSONL records uses < 5 MB peak RAM and leaves zero leaked file descriptors under fault injection.

### Card 3: The Python Data Model, Descriptors & Algorithmic Structures (Month 2)
- **Engineering Purpose:** Master object internals, attribute lookup, and O(1)/O(log N) data structures used in inference caches and schedulers.
- **Required Depth:** Level 4 (Implementation + Complexity Benchmarking).
- **Must Implement Yourself:** Dunder methods (`__repr__`, `__hash__`, `__eq__`, `__getitem__` slicing), `__slots__`, `weakref` registries, typed decorators, `__get__`/`__set__`/`__set_name__` descriptors, `FairRequestScheduler` (`heapq`), O(1) `LRUCache`, `TokenPrefixTrie`, and `mmap` `.safetensors` parser.
- **STOP WHEN:** Custom structures pass randomized property tests and empirical Big-O benchmarks.

### Card 4: Deep `asyncio`, Dynamic Batching, Wire Protocols & Multiprocessing (Month 3)
- **Engineering Purpose:** Build high-concurrency streaming servers and multi-process worker bridges without deadlocks, task leaks, or event-loop stalls.
- **Required Depth:** Level 4 (Systems Implementation).
- **Must Implement Yourself:** `TaskGroup` supervisors, cancellation/timeout shields, `AsyncDynamicBatcher`, length-prefixed TCP streams, `SSEParser`, raw ASGI streaming server with graceful drain, and `multiprocessing` (`spawn`) + `SharedMemory` zero-copy IPC.
- **STOP WHEN:** 100 concurrent streaming requests batch, stream, handle disconnects, and recover from simulated worker crashes cleanly.

### Card 5: Metaprogramming, AST, Bytecode & Framework Authoring (Month 5)
- **Engineering Purpose:** Demystify how PyTorch (`nn.Module`, `torch.compile`), Triton (`@triton.jit`), Pydantic, and FastAPI work internally.
- **Required Depth:** Level 4 (Framework Authoring from Scratch).
- **Must Implement Yourself:** `__init_subclass__` plugin registries, metaclasses, `MiniModule` parameter tracker, AST linter/transformer, `dis` bytecode analyzer, symbolic DAG tracer, `NanoSchema` (with codegen validator), and `NanoServe` (ASGI + DI + OpenAPI).
- **STOP WHEN:** Your hand-built `NanoServe` + `NanoSchema` framework powers your inference server with zero external web dependencies.

### Card 6: CPython Memory, Buffer Protocol, Strided Tensors, Autograd & Paged KV (Month 6)
- **Engineering Purpose:** Bridge high-level Python to low-level memory arenas, tensor strides, numerical quantization, and vLLM scheduling.
- **Required Depth:** Level 4 (From-Scratch Numerical & Engine Implementation).
- **Must Implement Yourself:** Zero-copy `memoryview` slab allocator, `StridedTensor` (zero-copy `transpose`/`broadcast_to`), `einsum`, IEEE-754 stable `softmax`, INT8/INT4 quantizer, `AutoTensor` autograd, `NanoLLM` (RoPE + GQA + KV Cache + BPE), and `ContinuousBatchScheduler` with PagedAttention Copy-on-Write & Prefix Caching.
- **STOP WHEN:** KV-cached decode matches full-sequence logits to `1e-5` and 1,000 continuous batching steps conserve 100% of physical KV blocks.

### Card 7: Profiling, Roofline, Free-Threading & Native C/C++/GPU Interop (Month 7)
- **Engineering Purpose:** Diagnose CPU/memory/GPU bottlenecks scientifically and cross the Python-to-C/CUDA boundary with zero copies.
- **Required Depth:** Level 4 (Systems Tuning & Native FFI).
- **Must Implement Yourself:** `MicroBench`, `cProfile`/`pstats` analyzer, `SamplingFlamegraphProfiler`, `AllocationHotspotAnalyzer`, `RooflineModel`, thread-safe PEP 703 structures, zero-copy `ctypes` C kernels (`fast_ops.c`), DLPack bridge, `MetaTensor` 0-byte verifier, and Tiled FlashAttention SPMD simulator.
- **STOP WHEN:** Every optimization has a 5-run before/after profile and native C kernels match reference outputs to `1e-5`.

### Card 8: Distributed IPC, Prometheus Telemetry & `Mini-vLLM` Capstone (Month 8)
- **Engineering Purpose:** Integrate all 8 months of Python mastery into a fault-tolerant, observable, multi-process inference engine control plane.
- **Required Depth:** Level 4 (Production Systems Capstone).
- **Must Implement Yourself:** `EngineCoreIPC`, `MultiRankWorkerGroup` (Tensor Parallelism simulation), `MetricsRegistry` (Prometheus text exposition), `CircuitBreaker`, `PrefixAwareLoadBalancer`, `IncrementalDetokenizer`, `JSONSchemaLogitsMasker`, `LoRACacheManager`, and the full `Mini-vLLM` capstone.
- **STOP WHEN:** Clean-room build, benchmark matrix, flamegraph profile, and 3 chaos post-mortems pass final review.

---

## 6. Authoritative Book Recommendations (by Author Name & Publisher Name)

Rather than juggling 15 scattered tutorials, use these **6 canonical engineering books** (4 Primary Core + 2 Specialized Reference). Every single week of the 8-month curriculum maps directly to exact chapters in these books:

### 6.1 Primary Core Books (Must-Have Shelf)

| Priority | Book Title & Edition | Author Name(s) | Publisher Name | Target Curriculum Phase & Why It Is Essential |
|---|---|---|---|---|
| **#1 Primary Bible (Base + Advanced)** | ***Fluent Python: Clear, Concise, and Effective Programming* (2nd Edition, 2022)** | **Luciano Ramalho** | **O'Reilly Media** | **Months 1, 2, 3 & 5 (Weeks 1–11, 17–20):** The undisputed gold standard for Python's Data Model (dunder methods, `__slots__`), type hints (`Protocol`, generics), closures, decorators, descriptors, iterators/generators, `asyncio`, concurrency, and metaprogramming (`__init_subclass__`, metaclasses). |
| **#2 Primary Typing & Architecture (Base)** | ***Robust Python: Write Clean and Maintainable Code* (1st Edition, 2021)** | **Patrick Viafore** | **O'Reilly Media** | **Months 1 & 4 (Weeks 1–4, 13–15):** Deep, practical coverage of strict static typing (`mypy`), `dataclass`, `Enum`, `Protocol`, `TypedDict`, custom collection types, context managers, and extensible design without inheritance bloat. |
| **#3 Primary Async & Streaming Systems (Base + Advanced)** | ***Python Concurrency with asyncio* (1st Edition, 2022)** | **Matthew Fowler** | **Manning Publications** | **Months 3 & 8 (Weeks 9–12, 29):** The best engineering book ever written on `asyncio` internals, event loops, Tasks, timeouts, cancellation, raw async sockets, streaming HTTP, queues, `multiprocessing` bridges, and avoiding event-loop stalls. |
| **#4 Primary Performance, Memory & C Interop (Advanced)** | ***High Performance Python: Practical Performant Programming for Humans* (2nd Edition, 2020)** | **Micha Gorelick and Ian Ozsvald** | **O'Reilly Media** | **Months 2, 6 & 7 (Weeks 7, 12, 21–28):** Masterclass on `cProfile`, line/memory profiling, Python memory layout, `array`/`memoryview` zero-copy buffers, vectorization, `multiprocessing`/`shared_memory`, the GIL, and compiling C/C++ extensions (`ctypes`, `cffi`, Cython/native). |

### 6.2 Specialized Reference Books (Internals, Testing & LLM Mechanics)

| Role | Book Title & Edition | Author Name(s) | Publisher Name | Target Curriculum Phase & Why It Is Essential |
|---|---|---|---|---|
| **CPython VM & Bytecode Reference** | ***CPython Internals: Your Guide to the Python 3 Interpreter* (1st Edition, 2021)** | **Anthony Shaw** | **Real Python (Dan Bader / Real Python Press)** | **Months 5 & 6 (Weeks 18, 21):** Explains how CPython parses ASTs, compiles bytecode (`dis`), executes the evaluation frame loop, manages `PyObject` reference counts, and runs the generational cyclic garbage collector. |
| **Testing Craftsmanship Reference** | ***Python Testing with pytest: Simple, Rapid, Effective, and Scalable* (2nd Edition, 2022)** | **Brian Okken** | **The Pragmatic Bookshelf (Pragmatic Programmers)** | **Months 1–4 (Week 3 & Daily Task 3):** Concise guide to writing professional, fast, parametrized `pytest` suites, fixtures, `tmp_path`, `monkeypatch`, and testing CLI/package boundaries. |
| **Clean Architecture Reference** | ***Architecture Patterns with Python: Enabling Test-Driven Development, Domain-Driven Design, and Event-Driven Microservices* (1st Edition, 2020)** | **Harry Percival and Bob Gregory** | **O'Reilly Media** | **Months 1 & 4 (Weeks 4, 13):** Shows how to separate pure domain models from infrastructure/I/O adapters using `Protocol` ports and lightweight dependency injection. |
| **Best-Practice Idioms Reference** | ***Effective Python: 125 Specific Ways to Write Better Python* (3rd Edition, 2024)** | **Brett Slatkin** | **Addison-Wesley Professional (Pearson)** | **Months 1–8 Daily Polish:** Actionable rules on modern Python 3.12/3.13 idioms, concurrency pitfalls, robustness, and API design. |
| **From-Scratch LLM Mechanics Bridge** | ***Build a Large Language Model (From Scratch)* (1st Edition, 2024)** | **Sebastian Raschka** | **Manning Publications** | **Month 6 (Week 23):** Clear reference for BPE tokenization, causal self-attention, RoPE, layer normalization, and autoregressive token generation loops. |

---

## 7. Value-Added Engineering Additions (What Most Prompts Miss)

To make sure your 8-month journey produces **real, senior-level engineering skill** rather than passive reading, this curriculum includes five critical systems mechanisms you might not have thought to ask for:

### 7.1 The Dual-Mode AI Engineering Protocol

Modern senior systems engineering and 2026+ technical interviews (Google Code Comprehension Round, Meta AI-Assisted Coding Interview, Anthropic 4-Level Progressive OA, and Stripe Bug Squash) evaluate two complementary competencies: the ability to build bulletproof systems from first principles without assistance, and the ability to steer AI assistants critically without succumbing to hallucinations or subtle bugs. This curriculum strictly divides your engineering time into two distinct operational modes:

#### Mode A: Blank-File Construction (Days 1–5 of Every Week & All of Month 1)
- **Inline AI autocomplete (Copilot/Cursor ghost-text) is STRICTLY DISABLED during Task 2.**
- Every class, type signature, algorithm, and test must be typed by hand from a blank buffer in `src/pymastery/`.
- **Why This Is Non-Negotiable:** Muscle memory for static type narrowing (`mypy --strict`), context management (`ExitStack`), memory layout (`memoryview`), and concurrency invariants only forms when your own nervous system struggles with and resolves compiler and runtime diagnostics. If an AI writes your type signatures or loops, you will fail unassisted live coding rounds and lack durable mental models.

#### Mode B: Adversarial AI Verification & Codebase Navigation (Day 6 of Weeks 5–32)
- From Month 2 onward, Day 6 shifts to practicing the exact "human-led, AI-assisted" workflow graded in top systems engineering interviews:
  - **Constrained Prompting:** Never ask vague, lazy prompts like *"Explain this file"* or *"Fix this bug"*. You must form an independent mental hypothesis first and write constrained, context-rich prompts (e.g., *"Trace how `allow_request()` mutates `self.tokens` across concurrent async tasks and check whether wall-clock drift in `time.time()` affects token refill math under leap seconds or NTP steps"*).
  - **Output Validation (Catching Hallucinations):** Deliberately audit AI explanations and AI-generated code for classic LLM blind spots:
    1. Using wall-clock `time.time()` instead of monotonic `time.monotonic()` for intervals, timeouts, and rate limits.
    2. Omitting `threading.Lock` or `asyncio.Lock` around shared mutable state.
    3. Swallowing `asyncio.CancelledError` inside blanket `except Exception:` blocks, breaking cancellation cascades.
    4. Using $O(N)$ `list.pop(0)` instead of $O(1)$ `collections.deque` in queues.
    5. Generating tautological unit tests that assert buggy behavior or over-mock internals with `unittest.mock.patch`.
    6. Overlooking unvalidated 64-bit integer lengths or out-of-bounds binary buffer offsets.
  - **Empirical Proof:** Never accept an AI claim at face value. You must write an empirical reproduction: a failing `pytest` or `tracemalloc` script that proves the AI's hallucination or concurrency bug, fix it yourself, and document the log in `artifacts/dayXXX/ai-audit.md`.

### 7.2 Your Standard `pyproject.toml` Quality Gate Template (Use from Day 1)
Every code file you write across all 224 days lives inside a single cohesive mono-package (`py-mastery/`) configured with strict verification from Day 1:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pymastery"
version = "0.1.0"
description = "From-scratch Systems Python & AI Inference Engineering Mastery"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.5",
]

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
disallow_any_unimported = true
disallow_any_generics = true

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF", "C90"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["error"]
```

### 7.3 The 7-Point Quality Code & Hiring Committee Rubric
Before checking off any day's final task box, evaluate your code against these 7 production and interview criteria:
1. **Types Tell the Whole Truth:** Could another engineer understand the inputs, outputs, and failure modes of your function from its signature alone without reading its body? Passes `mypy --strict` with zero `Any` and zero `# type: ignore`.
2. **Invalid States Are Unrepresentable:** Did you use `@dataclass(frozen=True, slots=True)`, `Enum`/`StrEnum`, and `TypeGuard`/`NewType` instead of passing untyped `dict`s or raw string/int flags?
3. **O(1) Memory & Deterministic Cleanup:** Does your code process streams lazily (`Iterator`/`AsyncIterator`) and close every file/socket/task/shared-memory block inside a `with`/`async with`/`finally` block? Verified with `tracemalloc`.
4. **Complexity Awareness:** Did you avoid hidden $O(N^2)$ string concatenations, `list.pop(0)`, or linear `x in list` scans inside hot loops?
5. **Failure-Tested:** Did you witness your test suite fail when you broke an invariant or ran the pre-seeded micro-bug, and pass when you restored it?
6. **Unfamiliar Code & Diff Hygiene:** Can you trace unfamiliar 300–500 line multi-file codebases in <20 minutes, isolate regressions with `git bisect run pytest`, and integrate features without breaking existing architectural contracts or introducing code churn?
7. **Hiring Committee Packet Narration & Adversarial AI Verification:** Can you state your clarifying assumptions, chosen pattern, time/space complexity, 10x load failure mode, and primary trade-off in 5 concise sentences, and empirically prove/disprove AI-generated suggestions?

### 7.4 How This Curriculum Connects Directly to Your AI Inference Engineering Curriculum
You do **not** have to wait 8 full months before touching `AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html`:
- **Option A (Sequential Foundation First):** Complete **Months 1–4 (Base Concepts, 1 hr/day)** first to achieve rock-solid Python & `asyncio` mastery, then start **Track 1 of the AI Inference Curriculum** while doing **Months 5–8 (Advanced Python, 1 hr/day)** in parallel! Every topic in Months 5–8 (tensors, KV cache, PagedAttention scheduler, profiling, C/CUDA interop, Prometheus) directly supercharges the corresponding weeks of the AI Inference Engineering Curriculum.
- **Option B (Parallel Daily Pair):** Spend **1 hour/day** on this Python Mastery Curriculum as your morning "warm-up & deep language craft" block right before your AI Inference Engineering lab block.

---

## 8. Final Mastery Checklists

### 8.1 Month 4 Base Concepts Gate Checklist (Day 112)
- [ ] I can create a clean `src/` Python package with `pyproject.toml`, `ruff`, `mypy --strict`, and `pytest` from memory.
- [ ] I write frozen, slotted dataclasses, `Protocol` interfaces, `Generic[T]` containers, and `ParamSpec` decorators without `Any`.
- [ ] I use `ExitStack`, custom context managers, chained exceptions, and `ExceptionGroup` so resources never leak on error.
- [ ] I build lazy O(1)-memory generator pipelines and verify peak heap memory with `tracemalloc`.
- [ ] I implement Python Data Model dunders (`__repr__`, `__eq__`, `__hash__`, `__getitem__` slicing), `__slots__`, `weakref`, and the descriptor protocol (`__get__`, `__set__`, `__set_name__`) from scratch.
- [ ] I have implemented an O(1) `LRUCache`, a `FairRequestScheduler` (`heapq`), a `TokenPrefixTrie`, and an `mmap` `.safetensors` binary reader by hand.
- [ ] I have implemented an `asyncio` `AsyncDynamicBatcher`, length-prefixed TCP RPC, an `SSEParser`, a raw ASGI streaming server with graceful drain, and a `multiprocessing.shared_memory` worker bridge (`pyserve-core`).
- [ ] I can navigate unfamiliar 300–500 line multi-file repositories cold, squash seeded bugs, and implement architectural feature extensions within 60 minutes.
- [ ] I can catch subtle AI hallucinations, tautological AI unit tests (over-mocking), and security vulnerabilities (path traversal, header bombs, Slowloris) during Mode B audits.
- [ ] I can execute 4-level progressive refactoring under shifting constraints (TTL expiration, atomic state rollback, async batch compaction) without ground-up rewrites.
- [ ] I can isolate regressions and resource leaks across git commit histories automatically using `git bisect run pytest`.

### 8.2 Month 8 Advanced Concepts Gate Checklist (Day 224)
- [ ] I can use `__init_subclass__`, metaclasses, `ast.NodeTransformer`, `dis` bytecode inspection, and `sys.monitoring` safely.
- [ ] I have built `NanoSchema` (with code-generated validators) and `NanoServe` (ASGI router + DI + SSE + OpenAPI) from scratch.
- [ ] I can explain CPython `PyObject` layout, refcounting, cyclic GC generations, `gc.freeze()`, and zero-copy `memoryview` slicing/casting.
- [ ] I have built a `StridedTensor` (with zero-copy `transpose` and `stride=0` broadcasting), an INT8/INT4 quantizer, a reverse-mode `AutoTensor` autograd engine, and a KV-cached Transformer (`NanoLLM` with RoPE, GQA, SwiGLU, RMSNorm, and BPE) from scratch.
- [ ] I have implemented a PagedAttention `PhysicalBlockAllocator` (with Copy-on-Write and Automatic Prefix Caching), `ContinuousBatchScheduler` (with chunked prefill and preemption), and speculative decoding verification loop in pure Python.
- [ ] I can profile CPU and memory using `cProfile`, stack-sampling flamegraphs, and roofline arithmetic-intensity models.
- [ ] I can bind native C kernels (`ctypes`/`cffi`/`pybind11` architecture) with zero-copy buffer pointers, verify 70B shapes in 0 MB with `MetaTensor`s, and explain DLPack, `@triton.jit`, and CUDA Graphs.
- [ ] I have built and defended the full `Mini-vLLM` multi-process capstone with Prometheus metrics, prefix-aware routing, circuit breakers, and chaos post-mortems.
- [ ] I can trace and modify actual production open-source `vllm` source files (`AsyncLLMEngine`, scheduler, incremental detokenizer, OpenAI serving protocol, and Prometheus metrics).
- [ ] I can audit C99 / C++ FFI bindings, fix pointer-lifetime and buffer-alignment bugs, pin `memoryview` buffers against GC, and release the GIL safely.
- [ ] I can wire a deterministic Agentic Tool-Call Execution Sandbox & Evaluation Harness validating emitted JSON tool arguments against `NanoSchema` under `asyncio.timeout`.
- [ ] I can deliver a crisp 2-minute oral Hiring Committee Packet Narration stating assumptions, architecture, Big-O bounds, 10x failure modes, and accepted design trade-offs.
"""

full_md = (
    HEADER_MD
    + build_days_markdown(
        WEEKS_PART1,
        "3. Phase 1 — Base Concepts Mastery (Months 1–4: Days 1–112)",
        "Every day below is a strictly bounded **1.0-hour engineering block** (15m reading, 30m from-scratch implementation, 15m testing/typing/failure diagnosis). By Day 112 (end of Month 4), you will have implemented every foundational and intermediate Python concept by hand to production quality.",
    )
    + build_days_markdown(
        WEEKS_PART2,
        "4. Phase 2 — Advanced Concepts Mastery (Months 5–8: Days 113–224)",
        "Phase 2 takes you from intermediate systems Python to **Senior/Staff AI Inference Python**: metaprogramming, AST/bytecode compilers, CPython memory layout, zero-copy buffer protocol, strided N-dimensional tensors, autograd, KV-cached Transformer execution, PagedAttention scheduling, profiling, native C/C++/GPU interop, and the full `Mini-vLLM` capstone.",
    )
    + FOOTER_MD
)

md_path.write_text(full_md, encoding="utf-8")
print(f"Wrote {md_path} ({len(full_md):,} bytes)")


def slugify(value: str) -> str:
    value = re.sub(r"`[^`]*`", lambda m: m.group(0)[1:-1], value)
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def format_inline(text: str) -> str:
    # Protect code spans first
    code_spans = []
    def repl_code(m):
        idx = len(code_spans)
        inner = html_lib.escape(m.group(1))
        if inner.startswith("[CORE: ") and inner.endswith("]"):
            code_spans.append(f'<span class="badge-core">{inner[1:-1]}</span>')
        elif inner.startswith("[MINOR: ") and inner.endswith("]"):
            code_spans.append(f'<span class="badge-minor">{inner[1:-1]}</span>')
        else:
            code_spans.append(f"<code>{inner}</code>")
        return f"@@CODE_{idx}@@"

    text = re.sub(r"`([^`]+)`", repl_code, text)
    text = html_lib.escape(text, quote=False)
    # Links
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # Bold + Italic
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    for idx, val in enumerate(code_spans):
        text = text.replace(f"@@CODE_{idx}@@", val)
    return text


def render_markdown_to_html(md_text: str):
    lines = md_text.splitlines()
    out = []
    toc_items = []
    i = 0
    n = len(lines)
    seen_slugs = {}

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(html_lib.escape(lines[i]))
                i += 1
            i += 1
            out.append(f'<pre><code class="language-{lang}">' + "\n".join(code_lines) + "</code></pre>")
            continue

        # Headings
        m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_head:
            level = len(m_head.group(1))
            raw_title = m_head.group(2).strip()
            base_slug = slugify(raw_title)
            count = seen_slugs.get(base_slug, 0)
            seen_slugs[base_slug] = count + 1
            slug = base_slug if count == 0 else f"{base_slug}_{count}"
            formatted_title = format_inline(raw_title)
            if level in (2, 3, 4):
                clean_toc = re.sub(r"<[^>]+>", "", formatted_title)
                indent = (level - 2) * 12
                toc_items.append(f'<a href="#{slug}" style="padding-left:{indent}px">{clean_toc}</a>')
            if 2 <= level <= 5:
                out.append(f'<h{level} id="{slug}"><span>{formatted_title}<a class="headerlink" href="#{slug}" title="Permanent link">&para;</a></span></h{level}>')
            else:
                out.append(f'<h{level} id="{slug}">{formatted_title}<a class="headerlink" href="#{slug}" title="Permanent link">&para;</a></h{level}>')
            i += 1
            continue

        # Blockquote / Callout
        if stripped.startswith(">"):
            q_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            q_text = "\n".join(q_lines).strip()
            m_call = re.match(r"^\[!(NOTE|TIP|WARNING|IMPORTANT)\]\s*(.*)$", q_text, flags=re.S)
            if m_call:
                kind = m_call.group(1).lower()
                paras = [p.strip() for p in m_call.group(2).split("\n\n") if p.strip()]
                inner_html = "".join(f"<p>{format_inline(p)}</p>" for p in paras)
                out.append(f'<aside class="callout callout-{kind}">{inner_html}</aside>')
            else:
                out.append(f"<blockquote><p>{format_inline(q_text)}</p></blockquote>")
            continue

        # Table
        if "|" in stripped and i + 1 < n and re.match(r"^\|?[\s:-]+\|[\s|:-]+$", lines[i + 1].strip()):
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i].strip() and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            thead = "<thead><tr>" + "".join(f"<th>{format_inline(h)}</th>" for h in headers) + "</tr></thead>"
            tbody_rows = []
            for r in rows:
                tbody_rows.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in r) + "</tr>")
            out.append(f"<table>{thead}<tbody>{''.join(tbody_rows)}</tbody></table>")
            continue

        # Unordered List
        if re.match(r"^-\s+", stripped):
            items = []
            while i < n and re.match(r"^-\s+", lines[i].strip()):
                item_text = re.sub(r"^-\s+", "", lines[i].strip())
                if item_text.startswith("[ ] "):
                    rest = format_inline(item_text[4:])
                    if rest.startswith("<strong>[CORE]"):
                        items.append(f'<li><label class="task-row core-task"><input class="task" type="checkbox"> {rest}</label></li>')
                    elif rest.startswith("<strong>[MINOR]"):
                        items.append(f'<li><label class="task-row minor-task"><input class="task" type="checkbox"> {rest}</label></li>')
                    else:
                        items.append(f'<li><label class="task-row"><input class="task" type="checkbox"> {rest}</label></li>')
                else:
                    items.append(f"<li>{format_inline(item_text)}</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>")
            continue

        # Paragraph
        p_lines = []
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s+|```|>|-\s+|\|)", lines[i].strip()):
            p_lines.append(lines[i].strip())
            i += 1
        out.append(f"<p>{format_inline(' '.join(p_lines))}</p>")

    body_html = "\n".join(out)
    body_html = re.sub(
        r'(?ms)(<h5 id="day-(\d+)[^>]*>.*?)(?=<h5 id="day-|<h[1-4][^>]*>|\Z)',
        lambda match: f'<section class="day-card" data-day="{match.group(2)}">{match.group(1)}</section>',
        body_html,
    )
    return body_html, "\n".join(toc_items)


body_html, toc_html = render_markdown_to_html(full_md)

ref_html = ref_html_path.read_text(encoding="utf-8")
css = re.search(r"<style>(.*?)</style>", ref_html, re.S).group(1)
base_script = re.search(r"<script>(.*?)</script>", ref_html, re.S).group(1)

custom_script = base_script.replace(
    "const key = 'ai-inference-curriculum-checks-v1';",
    "const key = 'python-mastery-curriculum-checks-v1';",
).replace(
    "const themeKey = 'ai-inference-curriculum-theme-v1';",
    "const themeKey = 'python-mastery-curriculum-theme-v1';",
)

week_options = "\n".join(
    [f'<option value="Month {m}">Month {m}</option>' for m in range(1, 9)]
    + [f'<option value="Week {w}">Week {w}</option>' for w in range(1, 33)]
    + ['<option value="6. Authoritative Book Recommendations">Book Recommendations</option>']
)

cover = """<section class="cover"><div class="kicker">From-Scratch Systems &amp; AI Inference Mastery</div><h1>Python 3.12+ Engineering Curriculum</h1><p>An 8-month (224-day, 1 hour/day) hands-on program to master production-grade Python from strict typing, data models, and async streaming to CPython internals, zero-copy buffers, strided tensors, native C/GPU interop, and building Mini-vLLM from scratch.</p><div class="meta">Months 1–4: Base Concepts (112 days) | Months 5–8: Advanced Concepts (112 days)<br>224 daily lessons | Strictly 1.0 focused hour per day (224 total hours)</div></section>"""

toolbar = f"""
<div class="toolbar">
    <div class="search-wrapper">
        <input id="search-input" aria-label="Search curriculum" placeholder="Find in curriculum (e.g. Day 112, asyncio, memoryview, PagedAttention)...">
        <span id="search-count" class="search-count"></span>
        <button id="search-prev" class="search-nav-btn" title="Previous match (Shift+Enter)" disabled>▲</button>
        <button id="search-next" class="search-nav-btn" title="Next match (Enter)" disabled>▼</button>
        <button id="search-clear" class="search-nav-btn" title="Clear search">✕</button>
    </div>
    <div class="filter-group" aria-label="Task filters">
        <button class="filter-btn active" data-filter="all">All Tasks</button>
        <button class="filter-btn" data-filter="core">Core Only</button>
        <button id="filter-rem" class="filter-btn" data-filter="remaining">Remaining Only</button>
    </div>
    <select id="week-jump" class="toolbar-select" aria-label="Jump to Month or Week">
        <option value="">Jump to Month / Week...</option>
        {week_options}
    </select>
    <button id="theme-toggle" class="toolbar-toggle" type="button">🌙 Dark</button>
    <button onclick="window.print()">Print / PDF</button>
    <button onclick="resetProgress()">Reset checks</button>
    <div class="progress-area" aria-live="polite">
        <span class="progress-label">0% Complete (0 / 0 Tasks)</span>
        <span class="progress-track"><span id="progress-fill" class="progress-fill"></span></span>
    </div>
</div>
"""

final_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Python 3.12+ Mastery Curriculum for AI Inference Engineering</title><style>{css}</style></head><body><div class="book">{cover}{toolbar}<div class="content"><nav class="toc" aria-label="Table of contents"><strong>Contents</strong>{toc_html}</nav><main>{body_html}</main></div></div><script>{custom_script}</script></body></html>"""

html_path.write_text(final_html, encoding="utf-8")
print(f"Wrote {html_path} ({len(final_html):,} bytes)")

browser_candidates = [
    shutil.which("msedge"),
    shutil.which("chrome"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
browser = next((c for c in browser_candidates if c and Path(c).exists()), None)
if browser:
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            str(html_path),
        ],
        check=True,
    )
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
else:
    print(f"PDF not generated: no Chromium browser found for {pdf_path}")
