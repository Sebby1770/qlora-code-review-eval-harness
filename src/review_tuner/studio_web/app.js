(function () {
  "use strict";

  const engine = window.ReviewEvalEngine;
  const state = {
    golden: [],
    predictions: [],
    predictionsB: null,
    result: null,
    compare: null,
    selected: 0,
    filterLang: "",
    filterSev: "",
  };

  const els = {
    status: document.getElementById("status"),
    goldenName: document.getElementById("golden-name"),
    predName: document.getElementById("pred-name"),
    predBName: document.getElementById("pred-b-name"),
    hero: document.getElementById("hero"),
    grade: document.getElementById("grade"),
    composite: document.getElementById("composite"),
    gate: document.getElementById("gate"),
    count: document.getElementById("count"),
    summary: document.getElementById("summary"),
    compareLine: document.getElementById("compare-line"),
    metrics: document.getElementById("metrics"),
    slices: document.getElementById("slices"),
    workspace: document.getElementById("workspace"),
    body: document.getElementById("example-body"),
    inspectTitle: document.getElementById("inspect-title"),
    inspectMeta: document.getElementById("inspect-meta"),
    inspectBody: document.getElementById("inspect-body"),
  };

  function setStatus(message, isError) {
    els.status.textContent = message;
    els.status.classList.toggle("error", Boolean(isError));
  }

  function fmt(value, digits) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(digits == null ? 3 : digits);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderDiff(diff) {
    const lines = String(diff).split("\n").map((line) => {
      let cls = "ctx";
      if (line.startsWith("+++") || line.startsWith("---")) cls = "file";
      else if (line.startsWith("+")) cls = "add";
      else if (line.startsWith("-")) cls = "del";
      else if (line.startsWith("@@")) cls = "hunk";
      return '<span class="' + cls + '">' + escapeHtml(line) + "</span>";
    });
    return '<pre class="diff">' + lines.join("\n") + "</pre>";
  }

  function metricCard(label, value) {
    return (
      '<article class="metric"><div class="value">' +
      escapeHtml(value) +
      '</div><div class="label">' +
      escapeHtml(label) +
      "</div></article>"
    );
  }

  function sliceCard(title, group) {
    const items = Object.keys(group)
      .sort()
      .map(
        (key) =>
          "<li><span>" +
          escapeHtml(key) +
          "</span><span class='value'>" +
          fmt(group[key]) +
          "</span></li>"
      )
      .join("");
    return (
      '<article class="slice-card"><div class="label">' +
      escapeHtml(title) +
      "</div><ul>" +
      items +
      "</ul></article>"
    );
  }

  function renderHero(aggregate) {
    els.hero.classList.remove("hidden");
    els.grade.textContent = aggregate.letter_grade;
    els.grade.className = "grade " + aggregate.letter_grade;
    els.composite.textContent = "composite " + fmt(aggregate.composite, 4);
    els.gate.textContent = aggregate.gate;
    els.gate.className = "chip " + aggregate.gate;
    els.count.textContent = aggregate.count + " examples";
    els.summary.textContent = aggregate.summary;
  }

  function renderMetrics(aggregate) {
    els.metrics.classList.remove("hidden");
    els.metrics.innerHTML = [
      metricCard("token F1", fmt(aggregate.token_f1)),
      metricCard("must-mention", fmt(aggregate.must_mention_recall)),
      metricCard("severity", fmt(aggregate.severity_accuracy)),
      metricCard("tag F1", fmt(aggregate.tag_f1)),
      metricCard("BLEU-lite", fmt(aggregate.bleu_lite)),
      metricCard("ROUGE-L", fmt(aggregate.rouge_l_lite)),
      metricCard("len ratio", fmt(aggregate.length_ratio)),
      metricCard("security-fail", fmt(aggregate.security_fail_rate)),
    ].join("");
  }

  function renderSlices(slices) {
    els.slices.classList.remove("hidden");
    els.slices.innerHTML = [
      sliceCard("Language", slices.by_language || {}),
      sliceCard("Severity", slices.by_severity || {}),
      sliceCard("Tag", slices.by_tag || {}),
      renderConfusion(state.result.aggregate.severity_confusion),
    ].join("");
  }

  function visibleDetails() {
    return state.result.details
      .map((row, index) => ({ row, index }))
      .filter(({ row }) => {
        if (state.filterLang && row.language !== state.filterLang) return false;
        if (state.filterSev && row.severity_expected !== state.filterSev) return false;
        return true;
      });
  }

  function renderFilters() {
    const lang = document.getElementById("filter-lang");
    const sev = document.getElementById("filter-sev");
    if (!lang || !sev || !state.result) return;
    const languages = Array.from(new Set(state.result.details.map((row) => row.language))).sort();
    const sevs = Array.from(new Set(state.result.details.map((row) => row.severity_expected))).sort();
    const keepLang = state.filterLang;
    const keepSev = state.filterSev;
    lang.innerHTML = '<option value="">all</option>' + languages.map((item) =>
      "<option value=\"" + escapeHtml(item) + "\">" + escapeHtml(item) + "</option>"
    ).join("");
    sev.innerHTML = '<option value="">all</option>' + sevs.map((item) =>
      "<option value=\"" + escapeHtml(item) + "\">" + escapeHtml(item) + "</option>"
    ).join("");
    lang.value = keepLang;
    sev.value = keepSev;
  }

  function waterfall(row) {
    const items = [
      ["token F1", row.token_f1],
      ["must-mention", row.must_mention_recall],
      ["severity", row.severity_accuracy],
      ["tag F1", row.tag_f1],
      ["allowed", 1 - (row.forbidden_rate || 0)],
    ];
    return (
      '<div class="waterfall">' +
      items
        .map((item) => {
          const pct = Math.max(0, Math.min(100, Math.round((item[1] || 0) * 100)));
          return (
            '<div class="wf-row"><span>' +
            escapeHtml(item[0]) +
            '</span><div class="wf-track"><span style="width:' +
            pct +
            '%"></span></div><em>' +
            fmt(item[1]) +
            "</em></div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function renderConfusion(matrix) {
    if (!matrix) return "";
    const labels = ["blocker", "high", "medium", "low", "nit"];
    let head = "<tr><th>exp \\ pred</th>" + labels.map((item) => "<th>" + item + "</th>").join("") + "</tr>";
    const body = labels
      .map((expected) => {
        const row = matrix[expected] || {};
        return (
          "<tr><th>" +
          expected +
          "</th>" +
          labels
            .map((predicted) => {
              const value = row[predicted] || 0;
              const cls = expected === predicted ? "ok" : value ? "missed" : "";
              return "<td class='" + cls + "'>" + value + "</td>";
            })
            .join("") +
          "</tr>"
        );
      })
      .join("");
    return '<article class="slice-card"><div class="label">Severity confusion</div><table class="confusion">' + head + body + "</table></article>";
  }

  function renderTable() {
    const details = visibleDetails();
    const compareMap = new Map();
    if (state.compare) {
      for (const row of state.compare.per_example) compareMap.set(row.id, row);
    }
    els.body.innerHTML = details
      .map(({ row, index }) => {
        const delta = compareMap.get(row.id);
        const deltaHtml = delta
          ? '<div class="hint">' + (delta.delta_composite >= 0 ? "+" : "") + fmt(delta.delta_composite) + "</div>"
          : "";
        return (
          '<tr data-index="' +
          index +
          '" class="' +
          (index === state.selected ? "active " : "") +
          (row.security_fail ? "security" : "") +
          '">' +
          "<td><code>" +
          escapeHtml(row.id) +
          "</code>" +
          deltaHtml +
          "</td>" +
          "<td>" +
          escapeHtml(row.language || "") +
          "</td>" +
          "<td>" +
          escapeHtml(row.severity_expected || "") +
          "</td>" +
          "<td>" +
          fmt(row.composite) +
          "</td>" +
          "<td>" +
          fmt(row.must_mention_recall) +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  function renderInspector() {
    const row = state.result.details[state.selected];
    if (!row) return;
    els.inspectTitle.textContent = row.id;
    els.inspectMeta.textContent = [row.language, row.severity_expected, row.file_path]
      .filter(Boolean)
      .join(" · ");
    const missed = row.missed_must_mention || [];
    const missedHtml = missed.length
      ? '<p class="missed">Missed must-mention: ' +
        missed.map((item) => "<code>" + escapeHtml(item) + "</code>").join(", ") +
        "</p>"
      : '<p class="ok">All required phrases mentioned.</p>';
    const compare = state.compare
      ? state.compare.per_example.find((item) => item.id === row.id)
      : null;
    const compareHtml = compare
      ? "<p>Compare Δ composite " +
        (compare.delta_composite >= 0 ? "+" : "") +
        fmt(compare.delta_composite) +
        " · caught A " +
        compare.caught_a +
        " → B " +
        compare.caught_b +
        "</p>"
      : "";
    els.inspectBody.innerHTML =
      "<h3>Diff</h3>" +
      renderDiff(row.diff || "") +
      '<div class="split">' +
      '<div class="pane-block"><h3>Expected</h3><pre>' +
      escapeHtml(row.expected_comment || "") +
      "</pre></div>" +
      '<div class="pane-block"><h3>Bot</h3><pre>' +
      escapeHtml(row.prediction || "") +
      "</pre></div>" +
      "</div>" +
      missedHtml +
      compareHtml +
      waterfall(row) +
      "<p class='hint'>token F1 " +
      fmt(row.token_f1) +
      " · BLEU " +
      fmt(row.bleu_lite) +
      " · ROUGE-L " +
      fmt(row.rouge_l_lite) +
      " · length " +
      fmt(row.length_ratio) +
      (row.security_fail ? " · security-fail" : "") +
      "</p>";
  }

  function renderAll() {
    if (!state.result) return;
    els.workspace.classList.remove("hidden");
    renderFilters();
    renderHero(state.result.aggregate);
    renderMetrics(state.result.aggregate);
    renderSlices(state.result.aggregate.slices);
    if (state.compare) {
      els.compareLine.classList.remove("hidden");
      els.compareLine.textContent =
        "Compare Δ composite " +
        (state.compare.delta_composite >= 0 ? "+" : "") +
        fmt(state.compare.delta_composite, 4) +
        " · newly caught " +
        state.compare.newly_caught.join(", ") +
        (state.compare.newly_caught.length ? "" : "(none)") +
        " · newly missed " +
        (state.compare.newly_missed.length ? state.compare.newly_missed.join(", ") : "(none)");
    } else {
      els.compareLine.classList.add("hidden");
    }
    renderTable();
    renderInspector();
  }

  function select(index) {
    const n = state.result ? state.result.details.length : 0;
    if (!n) return;
    state.selected = (index + n) % n;
    renderTable();
    renderInspector();
    const active = els.body.querySelector("tr.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }

  function scoreNow() {
    if (!state.golden.length) {
      setStatus("Load a golden JSONL first.", true);
      return;
    }
    if (!state.predictions.length) {
      setStatus("Load predictions, or score the heuristic baseline.", true);
      return;
    }
    try {
      state.result = engine.evaluateDataset(state.golden, state.predictions);
      state.compare = state.predictionsB
        ? engine.compareDatasets(state.golden, state.predictions, state.predictionsB)
        : null;
      state.selected = 0;
      setStatus("Scored " + state.result.aggregate.count + " examples in the browser.");
      renderAll();
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  }

  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error || new Error("read failed"));
      reader.readAsText(file);
    });
  }

  function loadGoldenText(text, label) {
    state.golden = engine.parseJsonl(text, engine.parseGolden);
    els.goldenName.textContent = label + " (" + state.golden.length + ")";
  }

  function loadPredText(text, label) {
    state.predictions = engine.parseJsonl(text, engine.parsePrediction);
    els.predName.textContent = label + " (" + state.predictions.length + ")";
  }

  function loadPredBText(text, label) {
    state.predictionsB = engine.parseJsonl(text, engine.parsePrediction);
    els.predBName.textContent = label + " (" + state.predictionsB.length + ")";
  }

  async function onFile(input, kind) {
    const file = input.files && input.files[0];
    if (!file) return;
    try {
      const text = await readFile(file);
      if (kind === "golden") loadGoldenText(text, file.name);
      else if (kind === "pred") loadPredText(text, file.name);
      else loadPredBText(text, file.name);
      if (state.golden.length && state.predictions.length) scoreNow();
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  }

  function bindDrop(label, input, kind) {
    label.addEventListener("dragover", (event) => {
      event.preventDefault();
      label.classList.add("hot");
    });
    label.addEventListener("dragleave", () => label.classList.remove("hot"));
    label.addEventListener("drop", async (event) => {
      event.preventDefault();
      label.classList.remove("hot");
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (!file) return;
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      await onFile(input, kind);
    });
    input.addEventListener("change", () => onFile(input, kind));
  }

  bindDrop(document.getElementById("golden-drop"), document.getElementById("golden-file"), "golden");
  bindDrop(document.getElementById("pred-drop"), document.getElementById("pred-file"), "pred");
  bindDrop(document.getElementById("pred-b-drop"), document.getElementById("pred-b-file"), "pred-b");

  async function loadSample() {
    const [goldenResp, predResp] = await Promise.all([
      fetch("samples/golden.jsonl", { cache: "no-store" }),
      fetch("samples/predictions.jsonl", { cache: "no-store" }),
    ]);
    if (!goldenResp.ok || !predResp.ok) {
      throw new Error("Could not fetch sample files. Serve the studio over HTTP.");
    }
    loadGoldenText(await goldenResp.text(), "sample golden");
    loadPredText(await predResp.text(), "sample predictions");
    state.predictionsB = null;
    els.predBName.textContent = "Second prediction file";
    scoreNow();
  }

  document.getElementById("sample-btn").addEventListener("click", async () => {
    try {
      await loadSample();
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  });

  document.getElementById("filter-lang").addEventListener("change", (event) => {
    state.filterLang = event.target.value;
    renderTable();
  });
  document.getElementById("filter-sev").addEventListener("change", (event) => {
    state.filterSev = event.target.value;
    renderTable();
  });
  document.getElementById("copy-summary").addEventListener("click", async () => {
    if (!state.result) {
      setStatus("Score a run first.", true);
      return;
    }
    try {
      await navigator.clipboard.writeText(state.result.aggregate.summary);
      setStatus("Summary copied.");
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  });
  document.getElementById("download-json").addEventListener("click", () => {
    if (!state.result) {
      setStatus("Score a run first.", true);
      return;
    }
    const blob = new Blob([JSON.stringify(state.result.aggregate, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "review-eval.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  if (!new URLSearchParams(location.search).has("fresh")) {
    loadSample().catch((err) => setStatus(String(err.message || err), true));
  }

  document.getElementById("baseline-btn").addEventListener("click", () => {
    if (!state.golden.length) {
      setStatus("Load a golden JSONL before running the heuristic baseline.", true);
      return;
    }
    state.predictions = state.golden.map((example) => engine.heuristicPrediction(example));
    els.predName.textContent = "heuristic baseline (" + state.predictions.length + ")";
    scoreNow();
  });

  els.body.addEventListener("click", (event) => {
    const row = event.target.closest("tr");
    if (!row) return;
    select(Number(row.getAttribute("data-index")));
  });

  document.addEventListener("keydown", (event) => {
    if (!state.result) return;
    const target = event.target;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
    if (event.key === "j" || event.key === "J") {
      event.preventDefault();
      select(state.selected + 1);
    } else if (event.key === "k" || event.key === "K") {
      event.preventDefault();
      select(state.selected - 1);
    }
  });
})();
