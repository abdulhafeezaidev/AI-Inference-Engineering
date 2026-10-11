from pathlib import Path
import re

ROOT = Path(__file__).parent
path = ROOT / 'AI_INFERENCE_ENGINEERING_CURRICULUM.md'
text = path.read_text(encoding='utf-8')
start_marker = '### Track 1 Weeks 1-8: granular daily curriculum'
end_marker = '### Track 1 capstone'
start = text.index(start_marker)
end = text.index(end_marker, start)
segment = text[start:end]

week_secondary = {
    1: ('Linux shell, files, and process signals', 'the shell and process tools are the workshop around the Python program'),
    2: ('Linux process inspection and GPU runtime checks', 'the operating environment must be visible and reproducible'),
    3: ('Numerical precision and tensor-shape checks', 'small shape and dtype mistakes can silently change model behavior'),
    4: ('Memory layout and per-token footprint analysis', 'capacity estimates connect model math to real GPU limits'),
    5: ('Container logs and engine startup diagnosis', 'logs tell you why an otherwise correct server did not start'),
    6: ('Benchmark metadata and statistical hygiene', 'good measurements need controlled conditions and honest summaries'),
    7: ('Hardware telemetry and timeline evidence', 'system tools turn a performance guess into a testable explanation'),
    8: ('Signals, logs, and service lifecycle', 'a useful endpoint must also stop, fail, and recover predictably'),
    9: ('Request IDs and operational log correlation', 'metrics show where a problem is and logs help explain one request'),
    10: ('Quality evaluation and reproducibility', 'a faster model is not useful if its answers fail the quality gate'),
    11: ('Queue metrics and overload behavior', 'the supporting measurements reveal when waiting, not execution, causes latency'),
    12: ('Evidence collection and incident writing', 'a failure becomes a learning artifact when its cause and recovery are recorded'),
    13: ('Version manifests and deployment hygiene', 'repeatable operations depend on knowing exactly what is running'),
    14: ('Architecture notes and experiment records', 'clear design records prevent capstone scope from expanding'),
    15: ('Prometheus evidence and resilience logs', 'optimization must remain observable and recoverable'),
    16: ('Documentation, presentation, and interview defense', 'career evidence is the bridge from a working project to a credible application'),
}
plain_by_week = {
    1: 'Think of this week as setting up a workshop. Python is the tool you will use to automate inference work, while Linux commands let you see what your programs are actually doing instead of guessing.',
    2: 'A container is a labeled box for the software and its dependencies. Linux and the NVIDIA runtime are the loading dock: they determine whether the box can actually reach the GPU, files, ports, and logs it needs.',
    3: 'A transformer is a chain of shape-preserving math operations. Treat tensor shapes like the dimensions written on pipes: if one pipe is the wrong size, the next operation either fails loudly or produces a misleading result.',
    4: 'Prefill is the model reading the prompt; decode is the model writing one new token at a time. The KV cache is like keeping notes from the pages already read so the model does not reread the whole book for every word.',
    5: 'An inference engine is the reusable machinery around a model. You are learning to operate one confidently, not to build every engine from scratch; the supporting logs and comparisons help you choose settings rationally.',
    6: 'A benchmark is an experiment, not a speed advertisement. Like timing a runner, you must warm up, repeat the run, control the course, and report slow as well as typical finishes.',
    7: 'Performance debugging is detective work. A GPU that looks idle may be waiting for the CPU, a copy, or a synchronization point, so the timeline is the evidence that tells you which suspect to test.',
    8: 'A service is more than a successful response. It must say whether it is ready, stream work, enforce deadlines, and leave no unfinished work behind when it shuts down.',
    9: 'Observability is the service speaking about itself. Counters tell you how much happened, gauges show what is happening now, and histograms show how a crowd of requests experienced latency.',
    10: 'Quantization packs numbers into fewer bits, like using a smaller measuring cup. It can save memory and improve speed, but you must check that the model still gives acceptable answers.',
    11: 'Batching is a shared bus for requests. Sharing the GPU can improve throughput, but long or numerous passengers can make everyone wait, so queue limits and fairness matter.',
    12: 'Chaos testing is a controlled fire drill. You break one safe part of the service, observe the signals, recover it, and record what would help an operator respond faster next time.',
    13: 'Production hardening is putting guardrails around a working machine. Version pins, probes, a runbook, and a cost model make the service repeatable for someone who did not build it.',
    14: 'The capstone starts with a contract, not a pile of features. A baseline is the ruler you use later to decide whether an optimization really helped.',
    15: 'Optimization is a before-and-after argument. You change one important thing, protect quality, watch the service, and use a profile to explain whether the bottleneck actually moved.',
    16: 'The final week turns engineering work into evidence another person can inspect. The goal is not a perfect system; it is a reproducible system whose tradeoffs and limits you can explain.',
}

def field(block, label, default=''):
    match = re.search(rf'^- \*\*{re.escape(label)}:\*\*\s*(.+)$', block, re.M)
    return match.group(1).strip() if match else default

def week_for(day):
    return (day - 1) // 7 + 1

def trap_for(title, week):
    low = title.lower()
    if any(word in low for word in ('async', 'timeout', 'cancel', 'stream', 'drain')):
        return 'Do not confuse a cancelled client with a completed request. Check task state and server logs, then verify cleanup rather than trusting the HTTP status alone.'
    if any(word in low for word in ('tensor', 'attention', 'embedding', 'logit', 'sampling', 'transformer')):
        return 'Do not rely on a printed value without checking shape, dtype, device, and tolerance. A tensor can look plausible while having the wrong dimension.'
    if any(word in low for word in ('kv', 'prefill', 'decode', 'memory', 'capacity')):
        return 'Do not compare runs with different prompt/output lengths or forget the factor of two for keys and values. Record units and assumptions beside every number.'
    if any(word in low for word in ('docker', 'container', 'gpu', 'nvidia')):
        return 'Do not assume a container can see the host GPU automatically. Verify the runtime, image, device visibility, mounted paths, and logs separately.'
    if any(word in low for word in ('benchmark', 'latency', 'throughput', 'percentile', 'roofline', 'profile', 'nsight')):
        return 'Do not report one fast run as a performance result. Exclude warm-up, repeat the test, keep the workload fixed, and record failures and percentiles.'
    if any(word in low for word in ('metric', 'prometheus', 'dashboard', 'observability')):
        return 'Do not put request IDs, prompts, or user data into metric labels. Keep labels bounded and use logs for per-request detail.'
    if any(word in low for word in ('quant', 'awq', 'gptq', 'fp8', 'quality')):
        return 'Do not call a quantized model better from memory or speed alone. Run the same quality set and keep incompatible-model failures visible.'
    if any(word in low for word in ('batch', 'queue', 'slo', 'backpressure', 'overload')):
        return 'Do not raise concurrency until throughput looks larger. Watch queue time, p95/p99, rejection rate, and fairness together.'
    if any(word in low for word in ('failure', 'chaos', 'oom', 'crash', 'post-mortem', 'restart')):
        return 'Inject faults only in the isolated test stack. Preserve logs before restarting, and never claim recovery without a successful post-fault smoke request.'
    if any(word in low for word in ('capstone', 'repository', 'runbook', 'cost', 'career', 'presentation', 'interview')):
        return 'Do not add features while a required artifact is still unverified. Freeze the scope, link every claim to evidence, and write down unknowns.'
    return f'Do not skip the verification step for {title.lower()}. A script that runs once is not yet a reliable engineering result.'

def secondary_task(secondary, week):
    commands = {
        1: 'Run `pwd`, `python --version`, `ps`, and `echo $?`; save the output beside the primary artifact.',
        2: 'Run `docker ps`, inspect the relevant process or container logs, and record GPU visibility or the exact missing prerequisite.',
        3: 'Add one assertion for shape/dtype/device and deliberately make it fail once before restoring it.',
        4: 'Write down units, dimensions, and one memory-footprint calculation beside the experiment.',
        5: 'Capture startup logs and one process/container status snapshot; label the symptom and likely layer.',
        6: 'Record hardware, software, model, workload, warm-up, repetition count, and failures in the run manifest.',
        7: 'Capture the relevant CPU/GPU/telemetry observation and state what evidence would falsify the primary hypothesis.',
        8: 'Send a signal or lifecycle event, inspect structured logs, and verify that no task or process is stranded.',
        9: 'Use a bounded request ID in logs and show the exact query or grep command that finds one request.',
        10: 'Run the same evaluation with fixed settings and save the baseline output before changing precision.',
        11: 'Record queue depth, wait time, status code, and p95 while changing arrival rate or concurrency.',
        12: 'Preserve logs, metrics, and timestamps before recovery; write one sentence describing the user impact.',
        13: 'Record image digest, Git commit, model revision, dependency lock, and runtime version.',
        14: 'Update the architecture or experiment note with one explicit assumption and one out-of-scope item.',
        15: 'Open the dashboard or metric output while the primary experiment runs and capture one correlated observation.',
        16: 'Link the artifact to the runbook or presentation and state one limitation honestly.',
    }
    return f'{secondary}: {commands[week]}'

def transform_block(day, title, block):
    week = week_for(day)
    secondary, _ = week_secondary[week]
    objective = field(block, 'Objective & Core Concepts', 'Complete the bounded topic for today.')
    mapped = field(block, 'Mapped Resource', 'Use the exact mapped resource in Section 7.')
    practical = field(block, 'Practical Task', 'Complete the practical task described by the objective.')
    deliverable = field(block, 'Deliverable', f'`artifacts/day{day:02d}.md`')
    stop = field(block, 'Stop Condition (STOP WHEN)', 'The listed test passes and the artifact is reproducible.')
    dont = field(block, 'Do Not Study Yet', 'Advanced internals and unrelated specializations.')
    note = plain_by_week[week]
    trap = trap_for(title, week)
    setup = f'Open the existing project and run `git status`; create a dated work folder such as `artifacts/day{day:02d}` before changing code.'
    verify = f'Run the project tests or the exact verification command from the practical task, then save the output next to {deliverable}.'
    diagnosis = f'Break one bounded assumption for `{title}` (invalid input, missing dependency, timeout, or wrong limit), observe the failure, restore it, and record the symptom.'
    return f'''##### Day {day}: {title} `[CORE: Track 1]` `[MINOR: {secondary}]`

> [!NOTE]
> **The Big Picture:** {note}
>
> Today the primary subject gets about 70% of the time. The secondary subject is the supporting lens that makes the primary task easier to operate and diagnose.

- **Primary Focus [CORE - 70% Weight]:** {objective}
- **Secondary Focus [MINOR - 30% Weight]:** {secondary}. {week_secondary[week][1]}.
- **Mapped Guides & Docs:** {mapped}

###### Step-by-Step To-Do Tasks:
- [ ] **[CORE] Task 1: Setup & Environment:** {setup}
- [ ] **[CORE] Task 2: Core Implementation:** {practical}
- [ ] **[CORE] Task 3: Verification & Test:** {verify}
- [ ] **[MINOR] Task 4: Systems / Tooling Task:** {secondary_task(secondary, week)}
- [ ] **Task 5: Diagnosis & Failure Test:** {diagnosis}

> [!WARNING]
> **Beginner Trap to Avoid:** {trap}

> [!IMPORTANT]
> **Daily boundary:** {stop}

- **Tangible Deliverable:** {deliverable}
- **STOP WHEN:** {stop}
- **Do Not Study Yet:** {dont}
- **Estimated Time:** 4.0 Hours total (Full Track: 4h | Compressed Track [Core Only]: 2.5h)

'''

heading_pattern = re.compile(r'(?m)^##### Day (\d+): ([^\n]+)\n')
headings = list(heading_pattern.finditer(segment))
if len(headings) != 112:
    raise SystemExit(f'Expected 112 day headings, found {len(headings)}')

parts = [segment[:headings[0].start()]]
for index, match in enumerate(headings):
    block_end = headings[index + 1].start() if index + 1 < len(headings) else len(segment)
    raw_block = segment[match.end():block_end]
    raw_title = match.group(2).strip()
    raw_title = re.sub(r'\s+`\[CORE:.*$', '', raw_title).strip()
    day = int(match.group(1))
    parts.append(transform_block(day, raw_title, raw_block))
new_segment = ''.join(parts)
path.write_text(text[:start] + new_segment + text[end:], encoding='utf-8')
print(f'Reformatted {len(headings)} Track 1 daily lessons with core/minor weighting and beginner guidance.')
