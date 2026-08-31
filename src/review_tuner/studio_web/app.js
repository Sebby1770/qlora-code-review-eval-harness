(() => {
  "use strict";

  const views = {
    home: document.getElementById("view-home"),
    guide: document.getElementById("view-guide"),
    grade: document.getElementById("view-grade"),
    run: document.getElementById("view-run"),
    compare: document.getElementById("view-compare"),
    dataset: document.getElementById("view-dataset"),
    how: document.getElementById("view-how"),
  };

  const HISTORY_KEY = "review-tuner-history";
  const THEME_KEY = "review-tuner-theme";

  const state = {
    lastView: null,
    lastCompare: null,
    inspectorId: null,
    lastPayload: null,
    sortKey: "composite",
    sortDir: 1,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function toast(message, kind) {
    const node = $("toast");
    if (!node) return;
    node.hidden = false;
    node.classList.toggle("error", kind === "error");
    node.textContent = message;
    window.clearTimeout(toast._timer);
    toast._timer = window.setTimeout(() => {
      node.hidden = true;
    }, 4200);
  }

  function route() {
    const hash = location.hash.replace(/^#/, "") || "/";
    const name = hash.split("?")[0].replace(/^\//, "") || "home";
    const known = views[name] ? name : "home";
    Object.entries(views).forEach(([key, node]) => {
      if (!node) return;
      node.hidden = key !== known;
    });
    document.querySelectorAll("[data-nav]").forEach((link) => {
      const active = link.getAttribute("data-nav") === known;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (known === "how") renderHow();
    if (known === "home") renderHistory();
    if (known === "run" && state.lastView) renderRun(state.lastView);
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    const text = await response.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch (err) {
      throw new Error("Studio returned a non-JSON response");
    }
    if (!response.ok) {
      throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
  }

  async function readFile(input) {
    const file = input.files && input.files[0];
    if (!file) throw new Error("Choose a file first.");
    return file.text();
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function escapeRegex(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function pct(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
  }

  function highlightComment(text, groups) {
    const source = String(text || "");
    const phrases = [];
    (groups || []).forEach((group) => {
      (group.phrases || []).forEach((phrase) => {
        if (phrase) phrases.push({ phrase: String(phrase), kind: group.kind });
      });
    });
    phrases.sort((a, b) => b.phrase.length - a.phrase.length);
    if (!phrases.length) return escapeHtml(source);
    const pattern = phrases.map((item) => escapeRegex(item.phrase)).join("|");
    const kindFor = {};
    phrases.forEach((item) => {
      kindFor[item.phrase.toLowerCase()] = item.kind;
    });
    const re = new RegExp(`(${pattern})`, "gi");
    let last = 0;
    let html = "";
    source.replace(re, (match, _g, offset) => {
      html += escapeHtml(source.slice(last, offset));
      const kind = kindFor[match.toLowerCase()] || "hit";
      html += `<mark class="${kind}">${escapeHtml(match)}</mark>`;
      last = offset + match.length;
      return match;
    });
    html += escapeHtml(source.slice(last));
    return html;
  }

  function renderDiff(diff) {
    const lines = String(diff || "").split("\n").map((line) => {
      let cls = "ctx";
      if (line.startsWith("+++") || line.startsWith("---")) cls = "meta";
      else if (line.startsWith("+")) cls = "add";
      else if (line.startsWith("-")) cls = "del";
      else if (line.startsWith("@@")) cls = "hunk";
      return `<div class="diff-line ${cls}">${escapeHtml(line) || "&nbsp;"}</div>`;
    });
    return `<pre class="diff">${lines.join("")}</pre>`;
  }

  function chips(items, kind) {
    if (!items || !items.length) return `<span class="hint">None</span>`;
    return items.map((item) => `<span class="chip ${kind}">${escapeHtml(item)}</span>`).join("");
  }

  function setStatus(id, message) {
    const node = $(id);
    if (node) node.textContent = message || "";
  }

  function setBusy(on) {
    document.querySelectorAll(".primary").forEach((btn) => {
      btn.disabled = Boolean(on);
      btn.classList.toggle("busy", Boolean(on));
    });
  }

  async function loadSampleGolden() {
    const data = await api("/api/samples/golden");
    return data.records;
  }

  async function loadSamplePredictions() {
    const data = await api("/api/samples/predictions");
    return data.records;
  }

  function persistRun(view) {
    state.lastView = view;
    try {
      if (state.lastPayload) {
        sessionStorage.setItem("review-tuner-last-payload", JSON.stringify(state.lastPayload));
      }
    } catch (err) {
      /* ignore */
    }
    try {
      const entry = {
        at: Date.now(),
        letter_grade: view.letter_grade,
        verdict: view.verdict,
        story: view.story,
        count: view.aggregate && view.aggregate.count,
      };
      const history = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
      history.unshift(entry);
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 5)));
    } catch (err) {
      /* ignore quota */
    }
    renderHistory();
  }

  function renderHistory() {
    const root = $("run-history");
    if (!root) return;
    let history = [];
    try {
      history = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
    } catch (err) {
      history = [];
    }
    if (!history.length) {
      root.innerHTML = `<p class="hint">No grades yet this session.</p>`;
      return;
    }
    root.innerHTML = history.map((item) => `
      <button type="button">
        <strong>${escapeHtml(item.letter_grade)} · ${escapeHtml(item.verdict)}</strong>
        <span class="hint"> ${escapeHtml(item.story || "")}</span>
      </button>`).join("");
  }

  async function gradePayload(payload) {
    state.lastPayload = payload;
    setStatus("grade-status", "Scoring…");
    setBusy(true);
    try {
      const view = await api("/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      persistRun(view);
      renderRun(view);
      location.hash = "#/run";
      setStatus("grade-status", "Done.");
      toast(`Graded ${view.aggregate.count} examples — letter ${view.letter_grade}.`);
      return view;
    } finally {
      setBusy(false);
    }
  }

  function renderRun(view) {
    const agg = view.aggregate || {};
    const letter = view.letter_grade || "F";
    const verdict = view.verdict || "Fail";
    const tone = verdict === "Pass" ? "pass" : verdict === "Weak" ? "weak" : "fail";
    $("run-hero").innerHTML = `
      <div class="letter ${tone}" aria-hidden="true">${escapeHtml(letter)}</div>
      <div>
        <h2>${escapeHtml(verdict)} — letter ${escapeHtml(letter)}</h2>
        <p>Overall ${pct(agg.composite)} across ${agg.count || 0} examples.
           Gate is ${pct(view.threshold)}.</p>
        <p class="meta">95% band ${pct(agg.composite_ci_lo)} – ${pct(agg.composite_ci_hi)}</p>
      </div>
    `;
    $("run-story").textContent = view.story || "";
    const slider = $("gate-slider");
    if (slider) {
      slider.value = String(Math.round(Number(view.threshold || 0.6) * 100));
      $("gate-value").textContent = pct(view.threshold || 0.6);
    }
    renderHistogram(view.histogram || {});
    renderByTag(view.by_tag || {});
    renderConfusion((view.error_analysis && view.error_analysis.severity_confusion) || []);
    const extra = view.extra_prediction_ids || [];
    const warning = $("run-warning");
    if (extra.length) {
      warning.hidden = false;
      warning.textContent = `Ignored extra predictions (not in the golden set): ${extra.join(", ")}`;
    } else {
      warning.hidden = true;
    }

    const worst = (view.examples || []).find((row) => row.id === view.worst_id) || (view.examples || [])[0];
    $("start-here").innerHTML = worst
      ? `<h2>Start here</h2>
         <p class="hint">Lowest overall score. Open it before you skim the table.</p>
         <p><strong>${escapeHtml(worst.id)}</strong> · ${escapeHtml(worst.file_path)} · ${escapeHtml(worst.verdict)}</p>
         <p>${escapeHtml(worst.expected_comment)}</p>
         <button class="ghost" type="button" data-open="${escapeHtml(worst.id)}">Inspect this example</button>`
      : "<p>No examples.</p>";

    const missed = (view.error_analysis && view.error_analysis.most_missed_must_mention) || [];
    $("missed-phrases").innerHTML = missed.length
      ? missed.map((row) => `<span class="chip miss">${escapeHtml(row.phrase)} ×${row.count}</span>`).join("")
      : `<span class="hint">No required phrases were skipped.</span>`;

    renderHeatmap(view.heatmap || { languages: [], severities: [], cells: [] });
    renderTable();
    renderMetrics(view);
    $("start-here").querySelector("[data-open]")?.addEventListener("click", () => {
      openInspector(worst.id);
    });
    if (state.inspectorId) openInspector(state.inspectorId);
    else $("inspector").hidden = true;
  }

  function heatColor(value) {
    const t = Math.max(0, Math.min(1, Number(value) || 0));
    const hue = 8 + t * 128;
    return `hsl(${hue} 52% ${72 - t * 18}%)`;
  }

  function renderHistogram(hist) {
    const root = $("histogram");
    if (!root) return;
    const counts = ["A", "B", "C", "D", "F"].map((letter) => Number(hist[letter] || 0));
    const max = Math.max(1, ...counts);
    root.innerHTML = ["A", "B", "C", "D", "F"].map((letter, index) => {
      const n = counts[index];
      const width = Math.round((n / max) * 100);
      return `<div class="hist-cell"><strong>${letter}</strong>${n}
        <div class="bar" aria-hidden="true"><span style="width:${width}%"></span></div></div>`;
    }).join("");
  }

  function renderConfusion(rows) {
    const root = $("confusion");
    if (!root) return;
    if (!rows.length) {
      root.innerHTML = "";
      return;
    }
    const body = rows.map((row) => `<tr>
      <td>${escapeHtml(row.predicted)}</td>
      <td>${escapeHtml(row.gold)}</td>
      <td class="num">${intOrZero(row.count)}</td>
    </tr>`).join("");
    root.innerHTML = `<table><thead><tr><th>Predicted</th><th>Gold</th><th>Count</th></tr></thead><tbody>${body}</tbody></table>`;
  }

  function intOrZero(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }

  function renderByTag(mapping) {
    const root = $("by-tag");
    if (!root) return;
    const entries = Object.entries(mapping || {});
    if (!entries.length) {
      root.innerHTML = "";
      return;
    }
    root.innerHTML = `<span class="hint">By tag</span> ` + entries.map(([tag, value]) =>
      `<span class="chip hit">${escapeHtml(tag)} ${pct(value)}</span>`
    ).join("");
  }

  function restamp(threshold) {
    const view = state.lastView;
    if (!view) return;
    const composite = Number(view.aggregate && view.aggregate.composite);
    view.threshold = threshold;
    view.verdict = composite >= threshold ? "Pass" : composite >= threshold * 0.75 ? "Weak" : "Fail";
    const story = String(view.story || "");
    view.story = story.replace(/^Overall \w+ —/, `Overall ${view.verdict.toLowerCase()} —`);
    $("gate-value").textContent = pct(threshold);
    const storyNode = $("run-story");
    if (storyNode) storyNode.textContent = view.story;
    const heading = document.querySelector("#run-hero h2");
    if (heading) heading.textContent = `${view.verdict} — letter ${view.letter_grade}`;
    const letter = document.querySelector("#run-hero .letter");
    if (letter) {
      const tone = view.verdict === "Pass" ? "pass" : view.verdict === "Weak" ? "weak" : "fail";
      letter.className = `letter ${tone}`;
    }
  }

  function countJsonl(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed) throw new Error("Paste is empty.");
    if (trimmed.startsWith("[")) {
      const payload = JSON.parse(trimmed);
      if (!Array.isArray(payload)) throw new Error("JSON array required.");
      return payload.length;
    }
    let count = 0;
    trimmed.split(/\n/).forEach((line) => {
      if (!line.trim()) return;
      JSON.parse(line);
      count += 1;
    });
    if (!count) throw new Error("No JSONL objects found.");
    return count;
  }

  function bindPastePreview(inputId, statusId) {
    const input = $(inputId);
    const status = $(statusId);
    if (!input || !status) return;
    input.addEventListener("input", () => {
      status.hidden = false;
      try {
        status.textContent = `${countJsonl(input.value)} records ready.`;
      } catch (err) {
        status.textContent = err.message;
      }
    });
  }

  function applyTheme(theme) {
    const light = theme === "light";
    document.body.classList.toggle("theme-light", light);
    const btn = $("theme-toggle");
    if (btn) {
      btn.setAttribute("aria-pressed", light ? "true" : "false");
      btn.textContent = light ? "Dark desk" : "Light paper";
    }
    try {
      localStorage.setItem(THEME_KEY, light ? "light" : "dark");
    } catch (err) {
      /* ignore */
    }
  }

  function renderHeatmap(heat) {
    const root = $("heatmap");
    const langs = heat.languages || [];
    const sevs = heat.severities || [];
    if (!langs.length || !sevs.length) {
      root.innerHTML = `<p class="hint">No heatmap for this run.</p>`;
      return;
    }
    const lookup = {};
    (heat.cells || []).forEach((cell) => {
      lookup[`${cell.language}||${cell.severity}`] = cell;
    });
    const header = `<div class="heat-row" style="grid-template-columns: 7rem repeat(${sevs.length}, 1fr)">
      <span></span>${sevs.map((sev) => `<span class="heat-lab">${escapeHtml(sev)}</span>`).join("")}
    </div>`;
    const rows = langs.map((lang) => {
      const cells = sevs.map((sev) => {
        const cell = lookup[`${lang}||${sev}`];
        if (!cell) return `<span class="heat-cell" style="background:#1b2a24;color:#8aa197">—</span>`;
        return `<span class="heat-cell" data-lang="${escapeHtml(lang)}" data-sev="${escapeHtml(sev)}" style="background:${heatColor(cell.composite)}" title="${escapeHtml(lang)} / ${escapeHtml(sev)}">
          ${pct(cell.composite)}
        </span>`;
      });
      return `<div class="heat-row" style="grid-template-columns: 7rem repeat(${sevs.length}, 1fr)">
        <span class="heat-lab">${escapeHtml(lang)}</span>${cells.join("")}
      </div>`;
    });
    root.innerHTML = header + rows.join("");
  }

  function visibleExamples() {
    const examples = (state.lastView && state.lastView.examples) || [];
    const lang = ($("filter-language") && $("filter-language").value.trim().toLowerCase()) || "";
    const sev = ($("filter-severity") && $("filter-severity").value.trim().toLowerCase()) || "";
    const tag = ($("filter-tag") && $("filter-tag").value.trim().toLowerCase()) || "";
    const needle = ($("filter-id") && $("filter-id").value.trim().toLowerCase()) || "";
    const failures = $("filter-failures") && $("filter-failures").checked;
    return examples.filter((row) => {
      if (lang && String(row.language || "").toLowerCase() !== lang) return false;
      if (sev && String(row.severity || "").toLowerCase() !== sev) return false;
      if (needle && !String(row.id || "").toLowerCase().includes(needle)) return false;
      if (tag) {
        const tags = (row.tags || []).map((item) => String(item).toLowerCase());
        if (!tags.includes(tag)) return false;
      }
      if (failures && row.verdict === "caught") return false;
      return true;
    });
  }

  function orderedExamples() {
    const key = state.sortKey || "composite";
    const dir = state.sortDir || 1;
    return visibleExamples().slice().sort((a, b) => {
      const left = a[key];
      const right = b[key];
      if (typeof left === "number" || typeof right === "number") {
        return (Number(left) - Number(right)) * dir;
      }
      return String(left || "").localeCompare(String(right || "")) * dir;
    });
  }

  function renderTable() {
    const rows = orderedExamples()
      .map((row) => `<tr data-id="${escapeHtml(row.id)}" tabindex="0">
        <td><code>${escapeHtml(row.id)}</code></td>
        <td>${escapeHtml(row.language)}</td>
        <td>${escapeHtml(row.severity)}</td>
        <td>${escapeHtml(row.verdict)}</td>
        <td>${escapeHtml(row.grade)}</td>
        <td class="num">${pct(row.must_mention_recall)}</td>
        <td class="num">${pct(row.composite)}</td>
      </tr>`)
      .join("");
    $("example-table").querySelector("tbody").innerHTML =
      rows || `<tr><td colspan="7">No examples match these filters.</td></tr>`;
  }

  function renderMetrics(view) {
    const agg = view.aggregate || {};
    const plain = view.metric_plain || {};
    const keys = [
      "composite",
      "must_mention_recall",
      "severity_accuracy",
      "tag_f1",
      "token_f1",
      "bleu_lite",
      "rouge_l",
      "forbidden_rate",
      "exact_match",
    ];
    const body = keys
      .filter((key) => key in agg)
      .map((key) => `<tr>
        <td><code>${escapeHtml(key)}</code></td>
        <td class="num">${Number(agg[key]).toFixed(4)}</td>
        <td>${escapeHtml(plain[key] || "")}</td>
      </tr>`)
      .join("");
    $("metric-table").innerHTML = `<table><thead><tr><th>Signal</th><th>Mean</th><th>In English</th></tr></thead><tbody>${body}</tbody></table>`;
  }

  function neighborId(id, delta) {
    const rows = orderedExamples();
    const index = rows.findIndex((row) => row.id === id);
    if (index < 0 || !rows.length) return null;
    const next = rows[index + delta];
    return next ? next.id : null;
  }

  function openInspector(id) {
    const example = ((state.lastView && state.lastView.examples) || []).find((row) => row.id === id);
    if (!example) return;
    state.inspectorId = id;
    document.querySelectorAll("#example-table tbody tr").forEach((row) => {
      row.classList.toggle("active", row.getAttribute("data-id") === id);
    });
    const explanation = (example.explanation || [])
      .map((row) => `<div class="bar-row">
        <span>${escapeHtml(row.plain)}</span>
        <div class="bar" aria-hidden="true"><span style="width:${Math.round((row.value || 0) * 100)}%"></span></div>
        <span class="num">${pct(row.value)}</span>
      </div>`)
      .join("");
    const expectedHtml = highlightComment(example.expected_comment, [
      { kind: "hit", phrases: example.must_mention },
    ]);
    const predictedHtml = highlightComment(example.prediction, [
      { kind: "hit", phrases: example.must_mention_hits },
      { kind: "miss", phrases: example.missed_must_mention },
      { kind: "ban", phrases: example.forbidden_hits },
    ]);
    $("inspector").hidden = false;
    $("inspector").innerHTML = `
      <div class="inspector-nav">
        <button class="ghost" type="button" data-nav-ex="-1">Previous</button>
        <button class="ghost" type="button" data-nav-ex="1">Next</button>
      </div>
      <h2 id="inspector-title">${escapeHtml(example.id)} · ${escapeHtml(example.file_path)}</h2>
      <p class="hint">${escapeHtml(example.context)}</p>
      <div class="inspector-grid">
        <div>
          <h3>Diff</h3>
          ${renderDiff(example.diff)}
        </div>
        <div>
          <h3>Expected comment</h3>
          <p class="comment">${expectedHtml}</p>
          <h3>Bot comment</h3>
          <p class="comment">${predictedHtml}</p>
          <h3>Required phrases found</h3>
          <p>${chips(example.must_mention_hits, "hit")}</p>
          <h3>Required phrases missed</h3>
          <p>${chips(example.missed_must_mention, "miss")}</p>
          <h3>Banned phrases</h3>
          <p>${chips(example.forbidden_hits, "ban")}</p>
          <h3>Shared words</h3>
          <p>${chips(example.overlap_tokens, "hit")}</p>
          <h3>Why this score</h3>
          <div class="waterfall">${explanation}</div>
        </div>
      </div>
    `;
    $("inspector").querySelectorAll("[data-nav-ex]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = neighborId(id, Number(btn.getAttribute("data-nav-ex")));
        if (next) openInspector(next);
      });
    });
    $("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
    const title = $("inspector-title");
    if (title) title.setAttribute("tabindex", "-1"), title.focus();
  }

  async function goldenFromForm() {
    const source = document.querySelector("input[name='golden-source']:checked").value;
    if (source === "sample") return "sample";
    if (source === "file") return await readFile($("golden-file"));
    return $("golden-paste").value;
  }

  async function predictionsFromForm() {
    const source = document.querySelector("input[name='pred-source']:checked").value;
    if (source === "baseline") return "baseline";
    if (source === "sample") return "sample";
    if (source === "file") return await readFile($("pred-file"));
    return $("pred-paste").value;
  }

  function toggleSource(name, fileId, pasteId) {
    document.querySelectorAll(`input[name='${name}']`).forEach((input) => {
      input.addEventListener("change", () => {
        const value = document.querySelector(`input[name='${name}']:checked`).value;
        if ($(fileId)) $(fileId).hidden = value !== "file";
        if ($(pasteId)) $(pasteId).hidden = value !== "paste";
      });
    });
  }

  async function renderHow() {
    const data = await api("/api/glossary");
    $("glossary").innerHTML = (data.glossary || [])
      .map((item) => `<article><h3>${escapeHtml(item.term)}</h3><p>${escapeHtml(item.plain)}</p></article>`)
      .join("");
    const weights = (state.lastView && state.lastView.weights) || [
      { key: "must_mention_recall", weight: 0.2, plain: "Required phrases actually mentioned" },
      { key: "token_f1", weight: 0.25, plain: "Shared words with the expected comment" },
      { key: "severity_accuracy", weight: 0.15, plain: "Correct seriousness label" },
      { key: "bleu_lite", weight: 0.1, plain: "Short phrase overlap" },
      { key: "rouge_l", weight: 0.1, plain: "Longest matching word sequence" },
      { key: "tag_f1", weight: 0.1, plain: "Issue tags" },
      { key: "length_ratio", weight: 0.05, plain: "Similar comment length" },
      { key: "forbidden_rate", weight: 0.05, plain: "Avoided banned phrases" },
    ];
    $("weights").innerHTML = weights
      .map((row) => `<div class="weight-row"><b>${Math.round(row.weight * 100)}%</b><span>${escapeHtml(row.plain)}</span></div>`)
      .join("");
  }

  function coverageHtml(coverage) {
    if (!coverage) return "";
    const block = (title, mapping) => {
      const entries = Object.entries(mapping || {});
      if (!entries.length) return `<div><h3>${escapeHtml(title)}</h3><p class="hint">None listed.</p></div>`;
      return `<div><h3>${escapeHtml(title)}</h3>${entries.map(([key, count]) =>
        `<span class="chip hit">${escapeHtml(key)} ×${count}</span>`).join(" ")}</div>`;
    };
    return `<div class="coverage">
      ${block("Languages", coverage.languages)}
      ${block("Severity", coverage.severities)}
      ${block("Tags", coverage.tags)}
    </div>`;
  }

  function renderCompare(report) {
    state.lastCompare = report;
    const delta = report.delta_b_minus_a || {};
    const rows = Object.entries(delta)
      .map(([key, value]) => {
        const cls = key === "forbidden_rate"
          ? (value < 0 ? "delta-good" : value > 0 ? "delta-bad" : "")
          : (value > 0 ? "delta-good" : value < 0 ? "delta-bad" : "");
        const sign = value > 0 ? "+" : "";
        return `<tr><td><code>${escapeHtml(key)}</code></td><td class="num ${cls}">${sign}${Number(value).toFixed(4)}</td></tr>`;
      })
      .join("");
    const flips = (list, title) => list.length
      ? `<h3>${title}</h3><ul>${list.map((row) =>
        `<li><button class="ghost" type="button" data-flip="${escapeHtml(row.id)}"><code>${escapeHtml(row.id)}</code> ${escapeHtml(row.from)} → ${escapeHtml(row.to)}</button></li>`
      ).join("")}</ul>`
      : `<h3>${title}</h3><p class="hint">None.</p>`;
    $("compare-out").innerHTML = `
      <article class="panel">
        <h2>${escapeHtml(report.a.letter_grade)} → ${escapeHtml(report.b.letter_grade)}</h2>
        <p class="hint">Positive composite / mention / tag numbers mean B is better. Forbidden-rate is the opposite. Click a flipped example to read both comments.</p>
        <table><thead><tr><th>Signal</th><th>B − A</th></tr></thead><tbody>${rows}</tbody></table>
        ${flips(report.improved || [], "Newly caught")}
        ${flips(report.regressed || [], "Newly missed")}
        <div id="compare-inspect"></div>
      </article>
    `;
    $("compare-out").querySelectorAll("[data-flip]").forEach((btn) => {
      btn.addEventListener("click", () => openCompareInspect(btn.getAttribute("data-flip")));
    });
  }

  function openCompareInspect(id) {
    const report = state.lastCompare;
    if (!report) return;
    const left = ((report.a_view && report.a_view.examples) || []).find((row) => row.id === id);
    const right = ((report.b_view && report.b_view.examples) || []).find((row) => row.id === id);
    const box = (label, example) => example
      ? `<div><h3>${escapeHtml(label)} · ${escapeHtml(example.verdict)} · ${pct(example.composite)}</h3>
         <p class="comment">${highlightComment(example.prediction, [
           { kind: "hit", phrases: example.must_mention_hits },
           { kind: "miss", phrases: example.missed_must_mention },
           { kind: "ban", phrases: example.forbidden_hits },
         ])}</p></div>`
      : `<p class="hint">${escapeHtml(label)} has no comment for this id.</p>`;
    const mount = $("compare-inspect");
    if (!mount) return;
    mount.innerHTML = `<div class="inspector-grid">${box("A", left)}${box("B", right)}</div>`;
  }

  function renderLint(result) {
    const issues = result.issues || [];
    const rows = issues
      .map((issue) => `<tr>
        <td>${escapeHtml(issue.level)}</td>
        <td><code>${escapeHtml(issue.rule)}</code></td>
        <td>${escapeHtml(issue.id || "")}</td>
        <td>${escapeHtml(issue.message)}</td>
      </tr>`)
      .join("");
    $("lint-out").innerHTML = `
      <article class="panel">
        <h2>${result.ok ? "Ready to grade" : "Fix the errors first"}</h2>
        <p>${result.errors} error(s), ${result.warnings} warning(s)${result.coverage ? ` · ${result.coverage.count} examples` : ""}.</p>
        ${coverageHtml(result.coverage)}
        <div class="table-wrap"><table>
          <thead><tr><th>Level</th><th>Rule</th><th>ID</th><th>Message</th></tr></thead>
          <tbody>${rows || "<tr><td colspan='4'>No issues.</td></tr>"}</tbody>
        </table></div>
      </article>
    `;
  }

  async function initHealth() {
    try {
      const health = await api("/api/health");
      $("health-pill").textContent = `v${health.version} · local · no GPU`;
    } catch (err) {
      $("health-pill").textContent = "Studio API unreachable";
    }
  }

  function typingTarget(event) {
    const tag = (event.target && event.target.tagName) || "";
    return tag === "INPUT" || tag === "TEXTAREA" || event.target.isContentEditable;
  }

  document.querySelectorAll("[data-action='sample']").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await gradePayload({ golden: "sample", predictions: "sample" });
      } catch (err) {
        toast(err.message, "error");
      }
    });
  });
  document.querySelectorAll("[data-action='baseline']").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await gradePayload({ golden: "sample", predictions: "baseline" });
      } catch (err) {
        toast(err.message, "error");
      }
    });
  });
  document.querySelectorAll("[data-action='goto-grade']").forEach((btn) => {
    btn.addEventListener("click", () => {
      location.hash = "#/grade";
    });
  });

  $("grade-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const form = event.target;
      await gradePayload({
        golden: await goldenFromForm(),
        predictions: await predictionsFromForm(),
        language: form.language.value || null,
        severity: form.severity.value || null,
        tag: form.tag.value || null,
        threshold: Number(form.threshold.value || 0.6),
      });
    } catch (err) {
      setStatus("grade-status", err.message);
      toast(err.message, "error");
    }
  });

  ["filter-language", "filter-severity", "filter-tag", "filter-id", "filter-failures"].forEach((id) => {
    $(id)?.addEventListener("input", renderTable);
    $(id)?.addEventListener("change", renderTable);
  });

  $("example-table")?.addEventListener("click", (event) => {
    const header = event.target.closest("th[data-sort]");
    if (header) {
      const key = header.getAttribute("data-sort");
      if (state.sortKey === key) state.sortDir *= -1;
      else {
        state.sortKey = key;
        state.sortDir = key === "composite" || key === "must_mention_recall" ? 1 : 1;
      }
      renderTable();
      return;
    }
    const row = event.target.closest("tr[data-id]");
    if (row) openInspector(row.getAttribute("data-id"));
  });
  $("heatmap")?.addEventListener("click", (event) => {
    const cell = event.target.closest("[data-lang]");
    if (!cell) return;
    if ($("filter-language")) $("filter-language").value = cell.getAttribute("data-lang") || "";
    if ($("filter-severity")) $("filter-severity").value = cell.getAttribute("data-sev") || "";
    renderTable();
  });
  $("example-table")?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("tr[data-id]");
    if (!row) return;
    event.preventDefault();
    openInspector(row.getAttribute("data-id"));
  });

  $("copy-story")?.addEventListener("click", async () => {
    const story = state.lastView && state.lastView.story;
    if (!story) return toast("Grade a run first.", "error");
    try {
      await navigator.clipboard.writeText(story);
      toast("Copied the English summary.");
    } catch (err) {
      toast("Could not copy — select the summary text instead.", "error");
    }
  });

  $("gate-slider")?.addEventListener("input", (event) => {
    restamp(Number(event.target.value) / 100);
  });

  $("theme-toggle")?.addEventListener("click", () => {
    applyTheme(document.body.classList.contains("theme-light") ? "dark" : "light");
  });

  $("download-badge")?.addEventListener("click", async () => {
    if (!state.lastPayload) return toast("Grade a run first.", "error");
    try {
      const response = await fetch("/api/badge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.lastPayload),
      });
      if (!response.ok) throw new Error("Could not build the badge.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "review-tuner-badge.svg";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("starter-row")?.addEventListener("click", () => {
    const radios = document.querySelectorAll("input[name='lint-source']");
    radios.forEach((input) => {
      input.checked = input.value === "paste";
    });
    if ($("lint-file")) $("lint-file").hidden = true;
    if ($("lint-paste")) {
      $("lint-paste").hidden = false;
      const starter = '{"id":"ex-new","language":"python","file_path":"app.py","context":"Describe the intended behaviour.","diff":"@@ example\\n-old\\n+new\\n","expected_comment":"Please restore the safety check and add a regression test.","severity":"high","tags":["security","tests"],"rubric":{"must_mention":["safety check","regression test"],"avoid":["style"]}}\n';
      $("lint-paste").value = ($("lint-paste").value + "\n" + starter).trim() + "\n";
      $("lint-paste").dispatchEvent(new Event("input"));
      $("lint-paste").focus();
    }
  });

  $("help-close")?.addEventListener("click", () => {
    $("help-overlay").hidden = true;
  });

  $("download-html")?.addEventListener("click", async () => {
    if (!state.lastPayload) return toast("Grade a run first.", "error");
    try {
      const response = await fetch("/api/report-html", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.lastPayload),
      });
      if (!response.ok) throw new Error("Could not build the HTML report.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "review-tuner-report.html";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("download-run")?.addEventListener("click", () => {
    if (!state.lastView) return toast("Grade a run first.", "error");
    const blob = new Blob([JSON.stringify(state.lastView, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `review-tuner-${state.lastView.letter_grade || "run"}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });

  $("compare-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      setBusy(true);
      const goldenSource = document.querySelector("input[name='cmp-golden']:checked").value;
      const golden = goldenSource === "sample" ? "sample" : await readFile($("cmp-golden-file"));
      const report = await api("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          golden,
          predictions_a: await readFile($("cmp-a-file")),
          predictions_b: await readFile($("cmp-b-file")),
        }),
      });
      renderCompare(report);
    } catch (err) {
      $("compare-out").innerHTML = `<p class="banner">${escapeHtml(err.message)}</p>`;
      toast(err.message, "error");
    } finally {
      setBusy(false);
    }
  });

  $("compare-sample")?.addEventListener("click", async () => {
    try {
      setBusy(true);
      const report = await api("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          golden: "sample",
          predictions_a: "baseline",
          predictions_b: "sample",
        }),
      });
      renderCompare(report);
    } catch (err) {
      $("compare-out").innerHTML = `<p class="banner">${escapeHtml(err.message)}</p>`;
      toast(err.message, "error");
    } finally {
      setBusy(false);
    }
  });

  $("lint-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const source = document.querySelector("input[name='lint-source']:checked").value;
      let body;
      if (source === "sample") body = { records: "sample" };
      else if (source === "file") body = { text: await readFile($("lint-file")) };
      else body = { text: $("lint-paste").value };
      renderLint(await api("/api/lint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }));
    } catch (err) {
      $("lint-out").innerHTML = `<p class="banner">${escapeHtml(err.message)}</p>`;
      toast(err.message, "error");
    }
  });

  document.addEventListener("keydown", (event) => {
    if (typingTarget(event)) return;
    if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
      event.preventDefault();
      $("help-overlay").hidden = !$("help-overlay").hidden;
      return;
    }
    if (event.key === "Escape") {
      if ($("help-overlay") && !$("help-overlay").hidden) {
        $("help-overlay").hidden = true;
        return;
      }
      if ($("inspector")) $("inspector").hidden = true;
      state.inspectorId = null;
      return;
    }
    const runVisible = views.run && !views.run.hidden;
    if (!runVisible || !state.lastView) return;
    if (event.key === "j" || event.key === "J") {
      const next = neighborId(state.inspectorId || orderedExamples()[0]?.id, 1);
      if (next) openInspector(next);
    }
    if (event.key === "k" || event.key === "K") {
      const rows = orderedExamples();
      const current = state.inspectorId || rows[0]?.id;
      const prev = neighborId(current, -1);
      if (prev) openInspector(prev);
    }
  });

  toggleSource("golden-source", "golden-file", "golden-paste");
  toggleSource("pred-source", "pred-file", "pred-paste");
  toggleSource("lint-source", "lint-file", "lint-paste");

  bindPastePreview("golden-paste", "golden-paste-status");
  bindPastePreview("pred-paste", "pred-paste-status");
  bindPastePreview("lint-paste", "lint-paste-status");
  try {
    applyTheme(localStorage.getItem(THEME_KEY) || "dark");
    const saved = sessionStorage.getItem("review-tuner-last-payload");
    if (saved) state.lastPayload = JSON.parse(saved);
  } catch (err) {
    applyTheme("dark");
  }

  window.addEventListener("hashchange", route);
  route();
  initHealth();
})();
