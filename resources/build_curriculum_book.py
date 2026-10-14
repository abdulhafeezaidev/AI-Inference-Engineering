from pathlib import Path
import re
import shutil
import subprocess
import markdown

ROOT = Path(__file__).parent
source = ROOT / 'AI_INFERENCE_ENGINEERING_CURRICULUM.md'
html_path = ROOT / 'AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.html'
pdf_path = ROOT / 'AI_INFERENCE_ENGINEERING_CURRICULUM_BOOK.pdf'

text = source.read_text(encoding='utf-8')
# Python-Markdown needs an explicit quoted blank line to keep adjacent GitHub
# alert blocks separate.
text = re.sub(r'\n\n> \[!(NOTE|TIP|WARNING|IMPORTANT)\]', r'\n>\n> [!\1]', text)
body = markdown.markdown(
    text,
    extensions=['tables', 'fenced_code', 'toc', 'attr_list', 'sane_lists'],
    extension_configs={'toc': {'title': 'Contents', 'permalink': True}},
)

# Give every checklist item a stable class so the browser can persist progress.
body = re.sub(r'<li>\s*(?:<p>)?\[ \]\s*', '<li><label><input class="task" type="checkbox"> ', body)
body = re.sub(r'<li>\s*(?:<p>)?\[x\]\s*', '<li><label><input class="task" type="checkbox" checked> ', body)
body = re.sub(r'(<li><label><input[^>]*> .*?)</li>', r'\1</label></li>', body)
body = re.sub(r'(<label><input class="task"[^>]*>\s*<strong>\[MINOR\])', r'<label class="task-row minor-task">\1', body)
body = re.sub(r'(<label><input class="task"[^>]*>\s*<strong>\[CORE\])', r'<label class="task-row core-task">\1', body)
body = re.sub(r'<code>\[CORE: ([^]]+)\]</code>', r'<span class="badge-core">CORE: \1</span>', body)
body = re.sub(r'<code>\[MINOR: ([^]]+)\]</code>', r'<span class="badge-minor">MINOR: \1</span>', body)
body = re.sub(r'<blockquote>\s*<p>\[!NOTE\]\s*(.*?)</p>\s*</blockquote>', r'<aside class="callout callout-note">\1</aside>', body, flags=re.S)
body = re.sub(r'<blockquote>\s*<p>\[!TIP\]\s*(.*?)</p>\s*</blockquote>', r'<aside class="callout callout-tip">\1</aside>', body, flags=re.S)
body = re.sub(r'<blockquote>\s*<p>\[!WARNING\]\s*(.*?)</p>\s*</blockquote>', r'<aside class="callout callout-warning">\1</aside>', body, flags=re.S)
body = re.sub(r'<blockquote>\s*<p>\[!IMPORTANT\]\s*(.*?)</p>\s*</blockquote>', r'<aside class="callout callout-important">\1</aside>', body, flags=re.S)
body = re.sub(
    r'(<aside class="callout callout-warning">.*?</p>)\s*<p>\[!IMPORTANT\]\s*(.*?)</aside>',
    r'\1</aside><aside class="callout callout-important">\2</aside>',
    body,
    flags=re.S,
)
body = re.sub(r'(<h[2-5][^>]*>)(.*?)(</h[2-5]>)', r'\1<span>\2</span>\3', body)
body = re.sub(
    r'(?ms)(<h5 id="day-(\d+)[^>]*>.*?)(?=<h5 id="day-|<h[1-4][^>]*>|\Z)',
    lambda match: f'<section class="day-card" data-day="{match.group(2)}">{match.group(1)}</section>',
    body,
)

css = r'''
@page { size: A4; margin: 18mm 16mm 20mm; @bottom-center { content: counter(page); color: #667085; font-size: 9pt; } }
:root { --ink:#172033; --muted:#667085; --navy:#172554; --blue:#2563eb; --cyan:#0891b2; --gold:#d97706; --mint:#e8f7f2; --paper:#f8fafc; --line:#d9e2ec; --rose:#fff1f2; }
[data-theme="dark"] { --ink:#e2e8f0; --muted:#94a3b8; --navy:#38bdf8; --blue:#38bdf8; --cyan:#22d3ee; --gold:#fbbf24; --mint:#14332d; --paper:#0b0f19; --line:#334155; --rose:#3b1d2a; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font-family: Georgia, 'Times New Roman', serif; font-size:11pt; line-height:1.55; transition:background .2s,color .2s; }
.book { max-width:1240px; margin:auto; background:#fff; min-height:100vh; }
[data-theme="dark"] .book { background:#111827; }
.cover { min-height: 720px; padding: 76px 70px; color:white; background: linear-gradient(135deg,#172554 0%,#075985 53%,#0f766e 100%); position:relative; overflow:hidden; }
.cover:after { content:''; position:absolute; width:430px; height:430px; right:-120px; bottom:-170px; border:2px solid rgba(255,255,255,.25); border-radius:50%; box-shadow:0 0 0 30px rgba(255,255,255,.07),0 0 0 60px rgba(255,255,255,.05); }
.kicker { text-transform:uppercase; letter-spacing:.14em; font:700 10pt Arial,sans-serif; color:#a5f3fc; }
.cover h1 { max-width:760px; margin:80px 0 20px; font:700 42pt Georgia,serif; line-height:1.06; }
.cover p { max-width:650px; font-size:15pt; color:#dbeafe; }
.cover .meta { position:absolute; left:70px; bottom:68px; font:11pt Arial,sans-serif; color:#cffafe; }
.toolbar { position:sticky; top:0; z-index:5; display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:12px 18px; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); font:10pt Arial,sans-serif; backdrop-filter:blur(12px); }
[data-theme="dark"] .toolbar { background:rgba(15,23,42,.96); }
.toolbar input { flex:1; min-width:160px; border:1px solid var(--line); border-radius:7px; padding:9px 11px; color:var(--ink); }
.toolbar button { border:0; border-radius:7px; padding:9px 12px; background:var(--navy); color:white; cursor:pointer; }
.toolbar button:hover { filter:brightness(1.08); }
.toolbar .progress { color:var(--ink); white-space:nowrap; font-weight:700; }
.toolbar-select, .toolbar-toggle { border:1px solid var(--line); border-radius:7px; padding:8px 10px; background:transparent; color:var(--ink); font:inherit; }
.filter-group { display:flex; gap:4px; }
.filter-btn { background:transparent !important; color:var(--ink) !important; border:1px solid var(--line) !important; padding:8px 10px !important; }
.filter-btn.active { background:var(--navy) !important; color:#fff !important; }
.progress-area { min-width:180px; flex:0 1 230px; }
.progress-label { display:block; margin-bottom:4px; color:var(--ink); font-weight:700; font-size:9pt; }
.progress-track { height:7px; overflow:hidden; border-radius:999px; background:#e2e8f0; }
.progress-fill { display:block; width:0; height:100%; border-radius:inherit; background:linear-gradient(90deg,#06b6d4,#2563eb,#8b5cf6); transition:width .25s ease; }
.content { display:grid; grid-template-columns:245px minmax(0,1fr); gap:32px; padding:30px 30px 70px; }
.toc { position:sticky; top:58px; align-self:start; max-height:calc(100vh - 80px); overflow:auto; padding:16px; background:#f1f5f9; border-left:4px solid var(--cyan); font:9pt Arial,sans-serif; }
.toc a { display:block; padding:4px 0; color:var(--navy); text-decoration:none; }
.toc a:hover { color:var(--blue); }
main { min-width:0; }
h1 { color:var(--navy); font-size:28pt; line-height:1.1; border-bottom:3px solid var(--cyan); padding-bottom:10px; }
h2 { margin-top:42px; padding:11px 14px; color:var(--navy); background:#e0f2fe; border-left:6px solid var(--blue); font-size:20pt; page-break-before:always; }
[data-theme="dark"] h2 { background:#172554; }
h2:first-of-type { page-break-before:auto; }
h3 { margin-top:30px; color:#075985; font-size:16pt; border-bottom:1px solid #bae6fd; padding-bottom:5px; }
h4 { color:#0f766e; font-size:13pt; }
h5 { margin:30px 0 10px; color:#172554; font-size:14pt; line-height:1.25; }
[data-theme="dark"] h3, [data-theme="dark"] h4, [data-theme="dark"] h5, [data-theme="dark"] strong { color:#bae6fd; }
p, ul, ol { max-width: 82ch; }
strong { color:#0f3b63; }
.badge-core, .badge-minor { display:inline-block; padding:3px 8px; border-radius:999px; font:700 8pt Arial,sans-serif; letter-spacing:.02em; vertical-align:middle; }
.badge-core { color:#fff; background:#172554; }
.badge-minor { color:#164e63; background:#ccfbf1; border:1px solid #99f6e4; }
[data-theme="dark"] .badge-core { color:#0b0f19; background:#38bdf8; }
[data-theme="dark"] .badge-minor { color:#ccfbf1; background:#164e63; border-color:#2dd4bf; }
table { border-collapse:collapse; width:100%; margin:18px 0 24px; font-size:9.3pt; page-break-inside:avoid; }
th { background:var(--navy); color:white; text-align:left; font-family:Arial,sans-serif; }
th, td { border:1px solid var(--line); padding:7px 8px; vertical-align:top; }
tr:nth-child(even) td { background:#f8fafc; }
[data-theme="dark"] td { background:#1e293b; border-color:#475569; }
[data-theme="dark"] tr:nth-child(even) td { background:#172033; }
blockquote { margin:18px 0; padding:12px 18px; border-left:5px solid var(--gold); background:#fffbeb; }
.callout { margin:18px 0; padding:14px 18px; border-left:6px solid; border-radius:0 8px 8px 0; page-break-inside:avoid; }
.callout { box-shadow:0 8px 24px rgba(15,23,42,.08); }
.callout-note { background:#ecfeff; border-color:#0891b2; }
.callout-tip { background:#ecfdf5; border-color:#059669; }
.callout-warning { background:#fffbeb; border-color:#d97706; }
.callout-important { background:#eef2ff; border-color:#4f46e5; }
pre { overflow:auto; padding:14px; background:#0f172a; color:#dbeafe; border-radius:7px; font:9pt Consolas,monospace; page-break-inside:avoid; }
code { color:#0f766e; font: .92em Consolas,monospace; }
pre code { color:inherit; }
li { margin:4px 0; }
.task { width:15px; height:15px; accent-color:var(--cyan); margin:0 8px 0 0; vertical-align:-2px; }
.task + strong { color:#0f3b63; }
.stop { background:var(--rose); border-left-color:#e11d48; }
mark.search-match { background:#fef08a !important; color:#172033 !important; padding:1px 3px; border-radius:3px; }
mark.search-match.active { background:#f97316 !important; color:#fff !important; font-weight:bold; box-shadow:0 0 0 3px rgba(249,115,22,.45); }
.search-wrapper { display:flex; align-items:center; background:#fff; border:1px solid var(--line); border-radius:8px; padding:3px 6px; flex:1; max-width:480px; }
.search-wrapper input { border:none !important; outline:none !important; flex:1; padding:6px 8px; font-size:10pt; }
.search-count { font-size:8.5pt; color:var(--muted); padding:0 8px; font-weight:600; white-space:nowrap; }
.search-nav-btn { background:transparent !important; color:var(--navy) !important; border:1px solid transparent !important; border-radius:4px; padding:4px 7px !important; cursor:pointer; font-size:9pt; }
.search-nav-btn:hover { background:#e2e8f0 !important; }
.search-nav-btn:disabled { opacity:.3; cursor:default; }
.day-card { scroll-margin-top:92px; }
[data-search-hidden="true"] { display:none !important; }
[data-theme="dark"] .toc { background:#1e293b; }
[data-theme="dark"] .toc a { color:#bae6fd; }
[data-theme="dark"] code { color:#67e8f9; }
@media print { body { background:white; } .toolbar, .toc { display:none; } .content { display:block; padding:0; } .cover { min-height:250mm; page-break-after:always; } a { color:inherit; text-decoration:none; } h2 { page-break-before:always; } }
@media (max-width: 800px) { .content { display:block; padding:16px; } .toc { position:relative; top:auto; max-height:260px; margin-bottom:20px; } .cover { min-height:560px; padding:42px 28px; } .cover h1 { margin-top:80px; font-size:31pt; } .cover .meta { left:28px; bottom:35px; } }
'''

script = r'''
const checks = [...document.querySelectorAll('.task')];
const key = 'ai-inference-curriculum-checks-v1';
const saved = JSON.parse(localStorage.getItem(key) || '{}');
checks.forEach((box, index) => { box.dataset.index = index; box.checked = !!saved[index]; box.addEventListener('change', () => { saved[index] = box.checked; localStorage.setItem(key, JSON.stringify(saved)); updateProgress(); }); });
const dayCards = [...document.querySelectorAll('.day-card')];
const taskRows = checks.map(box => box.closest('label')).filter(Boolean);
let activeFilter = 'all';
let activeQuery = '';

function updateProgress() {
    const done = checks.filter(x => x.checked).length;
    const percent = checks.length ? Math.round(done / checks.length * 100) : 0;
    document.querySelector('.progress-label').textContent = `${percent}% Complete (${done} / ${checks.length} Tasks)`;
    document.querySelector('.progress-fill').style.width = `${percent}%`;
}

function resetProgress() {
    localStorage.removeItem(key);
    checks.forEach(x => x.checked = false);
    updateProgress();
    applyView();
}

function applyView() {
    taskRows.forEach(row => {
        const box = row.querySelector('.task');
        const isMinor = row.textContent.includes('[MINOR]');
        const hiddenByFilter = (activeFilter === 'core' && isMinor) || (activeFilter === 'remaining' && box.checked);
        row.dataset.filterHidden = hiddenByFilter ? 'true' : 'false';
        row.style.display = hiddenByFilter ? 'none' : '';
    });
    dayCards.forEach(card => {
        const matchesSearch = !activeQuery || card.textContent.toLowerCase().includes(activeQuery.toLowerCase());
        const visibleTasks = [...card.querySelectorAll('.task')].some(box => {
            const row = box.closest('label');
            return row && row.dataset.filterHidden !== 'true';
        });
        const hideCard = !matchesSearch || (activeFilter === 'remaining' && !visibleTasks);
        card.dataset.searchHidden = hideCard ? 'true' : 'false';
    });
}

function setFilter(filter) {
    activeFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(button => button.classList.toggle('active', button.dataset.filter === filter));
    applyView();
}

document.querySelectorAll('.filter-btn').forEach(button => button.addEventListener('click', () => setFilter(button.dataset.filter)));

const themeKey = 'ai-inference-curriculum-theme-v1';
const themeButton = document.getElementById('theme-toggle');
function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(themeKey, theme);
    themeButton.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
}
setTheme(localStorage.getItem(themeKey) || 'light');
themeButton.addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

document.getElementById('week-jump').addEventListener('change', (event) => {
    const target = event.target.value;
    if (!target) return;
    const heading = [...document.querySelectorAll('main h2, main h3, main h4')].find(node => node.textContent.toLowerCase().includes(target.toLowerCase()));
    if (heading) heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
    event.target.value = '';
});
let currentMatches = [];
let currentMatchIndex = -1;
let searchDebounce = null;

const searchInput = document.getElementById('search-input');
const searchCount = document.getElementById('search-count');
const searchPrev = document.getElementById('search-prev');
const searchNext = document.getElementById('search-next');
const searchClear = document.getElementById('search-clear');

function clearHighlights() {
    document.querySelectorAll('mark.search-match').forEach(m => {
        const parent = m.parentNode;
        parent.replaceChild(document.createTextNode(m.textContent), m);
        parent.normalize();
    });
    currentMatches = [];
    currentMatchIndex = -1;
    searchCount.textContent = '';
    searchPrev.disabled = true;
    searchNext.disabled = true;
}

function highlightMatches(query) {
    clearHighlights();
    const q = query.trim();
    activeQuery = q;
    applyView();
    if (!q || q.length < 2) return;

    const main = document.querySelector('main');
    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            if (node.parentElement.closest('script, style, pre, code.nocopy')) return NodeFilter.FILTER_REJECT;
            return node.nodeValue.toLowerCase().includes(q.toLowerCase()) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
        }
    });

    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    const safeRegex = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${safeRegex})`, 'gi');

    nodes.forEach(node => {
        const text = node.nodeValue;
        if (regex.test(text)) {
            const span = document.createElement('span');
            span.innerHTML = text.replace(regex, '<mark class="search-match">$1</mark>');
            node.parentNode.replaceChild(span, node);
        }
    });

    currentMatches = [...document.querySelectorAll('mark.search-match')];
    if (currentMatches.length > 0) focusMatch(0);
    else searchCount.textContent = '0/0';
}

function focusMatch(index) {
    if (index < 0 || index >= currentMatches.length) return;
    currentMatches.forEach(m => m.classList.remove('active'));
    const active = currentMatches[index];
    active.classList.add('active');
    active.scrollIntoView({ behavior: 'smooth', block: 'center' });
    currentMatchIndex = index;
    searchCount.textContent = `${index + 1}/${currentMatches.length}`;
    searchPrev.disabled = false;
    searchNext.disabled = false;
}

function nextMatch() {
    if (currentMatches.length === 0) return;
    focusMatch((currentMatchIndex + 1) % currentMatches.length);
}

function prevMatch() {
    if (currentMatches.length === 0) return;
    focusMatch((currentMatchIndex - 1 + currentMatches.length) % currentMatches.length);
}

searchInput.addEventListener('input', (e) => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => highlightMatches(e.target.value), 200);
});

searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        if (e.shiftKey) prevMatch();
        else nextMatch();
    } else if (e.key === 'Escape') {
        searchInput.value = '';
        activeQuery = '';
        clearHighlights();
        applyView();
    }
});

searchNext.addEventListener('click', nextMatch);
searchPrev.addEventListener('click', prevMatch);
searchClear.addEventListener('click', () => {
    searchInput.value = '';
    activeQuery = '';
    clearHighlights();
    applyView();
    searchInput.focus();
});
updateProgress();
'''

cover = '''<section class="cover"><div class="kicker">A practical systems curriculum</div><h1>AI Inference Engineering</h1><p>A bounded, evidence-driven program from Python and Linux operations to model serving, observability, and production inference.</p><div class="meta">Track 1: 16 weeks of career-entry competence<br>112 daily lessons | 3-4 focused hours per day</div></section>'''
toolbar = '''
<div class="toolbar">
    <div class="search-wrapper">
        <input id="search-input" aria-label="Search curriculum" placeholder="Find in curriculum (e.g. Day 60, vLLM, Docker)...">
        <span id="search-count" class="search-count"></span>
        <button id="search-prev" class="search-nav-btn" title="Previous match (Shift+Enter)" disabled>▲</button>
        <button id="search-next" class="search-nav-btn" title="Next match (Enter)" disabled>▼</button>
        <button id="search-clear" class="search-nav-btn" title="Clear search">✕</button>
    </div>
    <div class="filter-group" aria-label="Task filters">
        <button class="filter-btn active" data-filter="all">All Tasks</button>
        <button class="filter-btn" data-filter="core">Core Only</button>
        <button class="filter-btn" data-filter="remaining">Remaining Only</button>
    </div>
    <select id="week-jump" class="toolbar-select" aria-label="Jump to week">
        <option value="">Jump to Week...</option>
        <option value="Week 1">Week 1</option><option value="Week 2">Week 2</option>
        <option value="Week 3">Week 3</option><option value="Week 4">Week 4</option>
        <option value="Week 5">Week 5</option><option value="Week 6">Week 6</option>
        <option value="Week 7">Week 7</option><option value="Week 8">Week 8</option>
        <option value="Week 9">Week 9</option><option value="Week 10">Week 10</option>
        <option value="Week 11">Week 11</option><option value="Week 12">Week 12</option>
        <option value="Week 13">Week 13</option><option value="Week 14">Week 14</option>
        <option value="Week 15">Week 15</option><option value="Week 16">Week 16</option>
        <option value="Track 2">Track 2</option>
    </select>
    <button id="theme-toggle" class="toolbar-toggle" type="button">🌙 Dark</button>
    <button onclick="window.print()">Print / PDF</button>
    <button onclick="resetProgress()">Reset checks</button>
    <div class="progress-area" aria-live="polite">
        <span class="progress-label">0% Complete (0 / 0 Tasks)</span>
        <span class="progress-track"><span id="progress-fill" class="progress-fill"></span></span>
    </div>
</div>
'''
html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Inference Engineering Curriculum</title><style>{css}</style></head><body><div class="book">{cover}{toolbar}<div class="content"><nav class="toc" aria-label="Table of contents"><strong>Contents</strong>{re.search(r'<div class="toc">(.*?)</div>', body, re.S).group(1) if '<div class="toc">' in body else '<p>Use search to navigate.</p>'}</nav><main>{body}</main></div></div><script>{script}</script></body></html>'''
html_path.write_text(html, encoding='utf-8')
print(f'Wrote {html_path}')

# Use a locally installed Chromium-family browser for print-to-PDF on Windows.
# HTML remains a successful build artifact when no browser is available.
browser_candidates = [
    shutil.which('msedge'),
    shutil.which('chrome'),
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
]
browser = next((candidate for candidate in browser_candidates if candidate and Path(candidate).exists()), None)
if browser:
    subprocess.run([
        browser,
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        f'--print-to-pdf={pdf_path}',
        str(html_path),
    ], check=True)
    print(f'Wrote {pdf_path}')
else:
    print(f'PDF not generated: install Microsoft Edge or Google Chrome; target is {pdf_path}')
