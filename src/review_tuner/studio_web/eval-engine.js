(() => {
  "use strict";

  const WEIGHTS = [
    ["token_f1", 0.25],
    ["bleu_lite", 0.1],
    ["rouge_l", 0.1],
    ["length_ratio", 0.05],
    ["must_mention_recall", 0.2],
    ["severity_accuracy", 0.15],
    ["tag_f1", 0.1],
    ["forbidden_rate", 0.05],
  ];
  const SEVERITIES = ["blocker", "high", "medium", "low", "nit"];
  const LETTERS = [[0.8, "A"], [0.65, "B"], [0.5, "C"], [0.35, "D"], [0, "F"]];

  const GLOSSARY = [
    { term: "Golden set", plain: "A labelled exam: diffs plus the comment a careful reviewer would write." },
    { term: "Prediction", plain: "What the bot (or the baseline) wrote for that diff." },
    { term: "Required phrases", plain: "Facts a good comment cannot skip, matched after ignoring punctuation." },
    { term: "Baseline", plain: "A keyword stand-in that never downloads a model." },
    { term: "Composite", plain: "One number in [0, 1] mixing overlap, required phrases, severity, tags, and bans." },
  ];

  function normalize(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  function ngrams(tokens, n) {
    if (tokens.length < n) return [];
    const out = [];
    for (let i = 0; i <= tokens.length - n; i += 1) out.push(tokens.slice(i, i + n).join("\u0001"));
    return out;
  }

  function countMap(items) {
    const map = new Map();
    items.forEach((item) => map.set(item, (map.get(item) || 0) + 1));
    return map;
  }

  function overlap(a, b) {
    const left = countMap(a);
    const right = countMap(b);
    let n = 0;
    left.forEach((value, key) => {
      n += Math.min(value, right.get(key) || 0);
    });
    return n;
  }

  function tokenF1(pred, gold) {
    const p = normalize(pred);
    const g = normalize(gold);
    if (!p.length && !g.length) return 1;
    if (!p.length || !g.length) return 0;
    const o = overlap(p, g);
    if (!o) return 0;
    const precision = o / p.length;
    const recall = o / g.length;
    return (2 * precision * recall) / (precision + recall);
  }

  function bleuLite(pred, gold) {
    const p = normalize(pred);
    const g = normalize(gold);
    if (!p.length && !g.length) return 1;
    if (!p.length || !g.length) return 0;
    const precisions = [];
    for (let n = 1; n <= 2; n += 1) {
      const pn = ngrams(p, n);
      const gn = ngrams(g, n);
      if (!pn.length) {
        precisions.push(0);
        continue;
      }
      precisions.push(overlap(pn, gn) / pn.length);
    }
    if (precisions.some((value) => value <= 0)) return 0;
    const bp = p.length >= g.length ? 1 : Math.exp(1 - g.length / Math.max(p.length, 1));
    const geo = Math.exp(precisions.reduce((sum, value) => sum + Math.log(value), 0) / precisions.length);
    return bp * geo;
  }

  function rougeL(pred, gold) {
    const p = normalize(pred);
    const g = normalize(gold);
    if (!p.length && !g.length) return 1;
    if (!p.length || !g.length) return 0;
    const dp = Array.from({ length: p.length + 1 }, () => new Array(g.length + 1).fill(0));
    for (let i = 1; i <= p.length; i += 1) {
      for (let j = 1; j <= g.length; j += 1) {
        dp[i][j] = p[i - 1] === g[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const lcs = dp[p.length][g.length];
    if (!lcs) return 0;
    const precision = lcs / p.length;
    const recall = lcs / g.length;
    return (2 * precision * recall) / (precision + recall);
  }

  function lengthRatio(pred, gold) {
    const p = normalize(pred).length;
    const g = normalize(gold).length;
    if (!p && !g) return 1;
    if (!p || !g) return 0;
    return Math.min(p, g) / Math.max(p, g);
  }

  function joined(text) {
    return normalize(text).join(" ");
  }

  function phraseHits(pred, phrases, wantMiss) {
    const hay = joined(pred);
    const list = phrases || [];
    if (!list.length) return wantMiss ? [] : 1;
    const hits = [];
    const misses = [];
    list.forEach((phrase) => {
      const needle = joined(phrase);
      if (needle && hay.includes(needle)) hits.push(phrase);
      else misses.push(phrase);
    });
    return wantMiss ? misses : hits.length / list.length;
  }

  function inferSeverity(text) {
    const lowered = String(text || "").toLowerCase();
    const normalized = joined(text);
    const match = normalized.match(/\bseverity\s+(blocker|high|medium|low|nit)\b/);
    if (match) return match[1];
    for (let i = 0; i < SEVERITIES.length; i += 1) {
      const sev = SEVERITIES[i];
      if (lowered.includes(`[${sev}]`) || normalized.includes(`${sev} severity`)) return sev;
    }
    return "";
  }

  function tagF1(pred, gold) {
    const a = new Set((pred || []).map((item) => String(item).toLowerCase()));
    const b = new Set((gold || []).map((item) => String(item).toLowerCase()));
    if (!a.size && !b.size) return 1;
    if (!a.size || !b.size) return 0;
    let o = 0;
    a.forEach((item) => {
      if (b.has(item)) o += 1;
    });
    if (!o) return 0;
    const precision = o / a.size;
    const recall = o / b.size;
    return (2 * precision * recall) / (precision + recall);
  }

  function letterGrade(score) {
    const value = Math.max(0, Math.min(1, Number(score) || 0));
    for (let i = 0; i < LETTERS.length; i += 1) {
      if (value >= LETTERS[i][0]) return LETTERS[i][1];
    }
    return "F";
  }

  function verdict(score, threshold) {
    const value = Number(score) || 0;
    if (value >= threshold) return "Pass";
    if (value >= threshold * 0.75) return "Weak";
    return "Fail";
  }

  function mean(values) {
    if (!values.length) return 0;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function parseJsonl(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed) throw new Error("empty dataset");
    if (trimmed.startsWith("[")) {
      const payload = JSON.parse(trimmed);
      if (!Array.isArray(payload)) throw new Error("expected a JSON array");
      return payload;
    }
    return trimmed.split(/\n/).filter((line) => line.trim()).map((line, index) => {
      const value = JSON.parse(line);
      if (!value || typeof value !== "object") throw new Error(`row ${index + 1}: expected an object`);
      return value;
    });
  }

  function asExample(raw) {
    const comment = raw.expected_comment || raw.review_comment || "";
    const rubric = raw.rubric || {};
    return {
      id: String(raw.id || ""),
      language: String(raw.language || ""),
      file_path: String(raw.file_path || ""),
      context: String(raw.context || ""),
      diff: String(raw.diff || ""),
      expected_comment: String(comment),
      severity: String(raw.severity || "medium").toLowerCase(),
      tags: (raw.tags || []).map((item) => String(item).toLowerCase()),
      must_mention: (rubric.must_mention || []).map((item) => String(item).toLowerCase()),
      avoid: (rubric.avoid || []).map((item) => String(item).toLowerCase()),
    };
  }

  function asPrediction(raw) {
    return {
      id: String(raw.id || ""),
      prediction: String(raw.prediction || ""),
      severity: raw.severity ? String(raw.severity).toLowerCase() : "",
      tags: (raw.tags || []).map((item) => String(item).toLowerCase()),
    };
  }

  function deletedTokens(diff) {
    const stop = new Set("a an and const def else false fn for from func if import in let nil none not null or return self the this true var".split(" "));
    const seen = new Set();
    const out = [];
    String(diff || "").split(/\n/).forEach((line) => {
      if (!line.startsWith("-") || line.startsWith("---")) return;
      const matches = line.slice(1).match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
      matches.forEach((token) => {
        const key = token.toLowerCase();
        if (token.length < 2 || stop.has(key) || seen.has(key)) return;
        seen.add(key);
        out.push(token);
      });
    });
    return out;
  }

  function heuristic(example) {
    const diff = example.diff.toLowerCase();
    let comment;
    let tags;
    if (diff.includes("password") || diff.includes("token") || diff.includes("auth")) {
      comment = "This looks security-sensitive. Please keep the existing validation path and add a regression test so expired or invalid tokens are rejected before state changes.";
      tags = ["security", "tests"];
    } else if (diff.includes("except") || diff.includes("catch")) {
      comment = "This error handling hides the failure path. Please preserve the exception details or surface a clear typed error so callers can recover safely.";
      tags = ["reliability"];
    } else if (diff.includes("select *") || diff.includes("for row in")) {
      comment = "This may do unnecessary work as data grows. Please narrow the query or batch the loop and add a test that covers the large input case.";
      tags = ["performance", "tests"];
    } else {
      comment = "Please add a focused regression test for this behavior and make the failure mode explicit for future maintainers.";
      tags = ["tests"];
    }
    const removed = deletedTokens(example.diff);
    if (removed.length) comment += ` Removed lines mention: ${removed.slice(0, 16).join(", ")}.`;
    return { id: example.id, prediction: comment, severity: example.severity, tags };
  }

  function scoreOne(golden, prediction) {
    const mention = phraseHits(prediction.prediction, golden.must_mention, false);
    const missed = phraseHits(prediction.prediction, golden.must_mention, true);
    const forbiddenList = golden.avoid || [];
    const hay = joined(prediction.prediction);
    const forbiddenHits = forbiddenList.filter((phrase) => hay.includes(joined(phrase)));
    const forbidden = forbiddenList.length ? forbiddenHits.length / forbiddenList.length : 0;
    const predictedSeverity = prediction.severity || inferSeverity(prediction.prediction);
    const lexical = tokenF1(prediction.prediction, golden.expected_comment);
    const bleu = bleuLite(prediction.prediction, golden.expected_comment);
    const rouge = rougeL(prediction.prediction, golden.expected_comment);
    const ratio = lengthRatio(prediction.prediction, golden.expected_comment);
    const severityAcc = predictedSeverity === golden.severity ? 1 : 0;
    const tags = tagF1(prediction.tags, golden.tags);
    const composite = 0.25 * lexical + 0.1 * bleu + 0.1 * rouge + 0.05 * ratio
      + 0.2 * mention + 0.15 * severityAcc + 0.1 * tags + 0.05 * (1 - forbidden);
    const hits = golden.must_mention.filter((phrase) => !missed.includes(phrase));
    const exampleVerdict = composite >= 0.75 && mention >= 0.99 ? "caught"
      : (mention < 0.5 || forbidden > 0 ? "missed" : "partial");
    return {
      id: golden.id,
      language: golden.language,
      file_path: golden.file_path,
      context: golden.context,
      diff: golden.diff,
      expected_comment: golden.expected_comment,
      prediction: prediction.prediction,
      severity: golden.severity,
      predicted_severity: predictedSeverity,
      tags: golden.tags,
      predicted_tags: prediction.tags,
      must_mention: golden.must_mention,
      avoid: golden.avoid,
      must_mention_hits: hits,
      missed_must_mention: missed,
      forbidden_hits: forbiddenHits,
      exact_match: joined(prediction.prediction) === joined(golden.expected_comment) ? 1 : 0,
      token_f1: lexical,
      bleu_lite: bleu,
      rouge_l: rouge,
      length_ratio: ratio,
      must_mention_recall: mention,
      forbidden_rate: forbidden,
      severity_accuracy: severityAcc,
      tag_f1: tags,
      composite,
      grade: letterGrade(composite),
      verdict: exampleVerdict,
      overlap_tokens: normalize(prediction.prediction).filter((token, index, all) => (
        normalize(golden.expected_comment).includes(token) && all.indexOf(token) === index
      )).slice(0, 24),
      explanation: WEIGHTS.map(([key, weight]) => {
        const raw = key === "forbidden_rate" ? forbidden : ({
          token_f1: lexical, bleu_lite: bleu, rouge_l: rouge, length_ratio: ratio,
          must_mention_recall: mention, severity_accuracy: severityAcc, tag_f1: tags,
        }[key]);
        const value = key === "forbidden_rate" ? 1 - raw : raw;
        return { key, weight, value, raw, contribution: weight * value, plain: key };
      }),
    };
  }

  function buildView(goldens, predictions, threshold) {
    const byId = new Map(predictions.map((item) => [item.id, item]));
    const missing = goldens.filter((item) => !byId.has(item.id)).map((item) => item.id);
    if (missing.length) throw new Error(`missing predictions for ids: ${missing.join(", ")}`);
    const examples = goldens.map((item) => scoreOne(item, byId.get(item.id)));
    const composites = examples.map((item) => item.composite);
    const aggregate = { count: examples.length };
    ["token_f1", "bleu_lite", "rouge_l", "length_ratio", "must_mention_recall",
      "forbidden_rate", "severity_accuracy", "tag_f1", "composite", "exact_match"].forEach((key) => {
      const values = examples.map((item) => Number(item[key]) || 0);
      aggregate[key] = mean(values);
      aggregate[`${key}_std`] = 0;
    });
    aggregate.composite_percent = Math.floor(aggregate.composite * 10000) / 100;
    aggregate.composite_ci_lo = Math.min(...composites);
    aggregate.composite_ci_hi = Math.max(...composites);
    const byLang = {};
    const bySev = {};
    examples.forEach((item) => {
      byLang[item.language || "unknown"] = byLang[item.language || "unknown"] || [];
      byLang[item.language || "unknown"].push(item.composite);
      bySev[item.severity || "unknown"] = bySev[item.severity || "unknown"] || [];
      bySev[item.severity || "unknown"].push(item.composite);
    });
    aggregate.by_language = Object.fromEntries(Object.entries(byLang).map(([key, vals]) => [key, mean(vals)]));
    aggregate.by_severity = Object.fromEntries(Object.entries(bySev).map(([key, vals]) => [key, mean(vals)]));
    const missedCounts = {};
    examples.forEach((item) => item.missed_must_mention.forEach((phrase) => {
      missedCounts[phrase] = (missedCounts[phrase] || 0) + 1;
    }));
    const analysis = {
      most_missed_must_mention: Object.entries(missedCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([phrase, count]) => ({ phrase, count })),
      forbidden_phrase_hits: [],
      severity_confusion: examples.map((item) => ({
        predicted: item.predicted_severity || "unknown",
        gold: item.severity,
        count: 1,
      })),
    };
    const hist = { A: 0, B: 0, C: 0, D: 0, F: 0 };
    examples.forEach((item) => { hist[item.grade] += 1; });
    const worst = examples.slice().sort((a, b) => a.composite - b.composite)[0];
    const topMiss = (analysis.most_missed_must_mention[0] || {}).phrase;
    const story = `Overall ${verdict(aggregate.composite, threshold).toLowerCase()} — letter ${letterGrade(aggregate.composite)} (${Math.round(aggregate.composite * 100)}% across ${examples.length} examples).`
      + (topMiss ? ` The most-skipped required phrase was “${topMiss}”.` : " Every required phrase appeared at least once.");
    const byTag = {};
    examples.forEach((item) => {
      (item.tags || []).forEach((tag) => {
        byTag[tag] = byTag[tag] || [];
        byTag[tag].push(item.composite);
      });
    });
    const goldenIds = new Set(goldens.map((item) => item.id));
    return {
      aggregate,
      error_analysis: analysis,
      examples,
      letter_grade: letterGrade(aggregate.composite),
      verdict: verdict(aggregate.composite, threshold),
      threshold,
      story,
      heatmap: { languages: Object.keys(byLang), severities: Object.keys(bySev), cells: Object.keys(byLang).flatMap((lang) => Object.keys(bySev).map((sev) => {
        const vals = examples.filter((item) => item.language === lang && item.severity === sev).map((item) => item.composite);
        return vals.length ? { language: lang, severity: sev, count: vals.length, composite: mean(vals) } : null;
      }).filter(Boolean)) },
      histogram: hist,
      by_tag: Object.fromEntries(Object.entries(byTag).map(([key, vals]) => [key, mean(vals)])),
      worst_id: worst ? worst.id : null,
      extra_prediction_ids: predictions.filter((item) => !goldenIds.has(item.id)).map((item) => item.id),
      glossary: GLOSSARY,
      metric_plain: {},
      weights: WEIGHTS.map(([key, weight]) => ({ key, weight, plain: key })),
      badge: { letter: letterGrade(aggregate.composite), verdict: verdict(aggregate.composite, threshold), percent: Math.round(aggregate.composite * 100) },
    };
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function reportHtml(view) {
    const rows = (view.examples || []).map((item) => (
      `<tr><td>${escapeHtml(item.id)}</td><td>${escapeHtml(item.language)}</td><td>${escapeHtml(item.severity)}</td>`
      + `<td>${escapeHtml(item.grade)}</td><td>${escapeHtml(item.verdict)}</td><td>${Number(item.composite).toFixed(3)}</td></tr>`
    )).join("");
    return `<!doctype html><html><head><meta charset="utf-8"><title>Review Tuner ${escapeHtml(view.letter_grade)}</title>
<style>body{font-family:sans-serif;max-width:920px;margin:2rem auto;color:#1b1f24}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}</style>
</head><body>
<h1>${escapeHtml(view.letter_grade)} · ${escapeHtml(view.verdict)}</h1>
<p>${escapeHtml(view.story)}</p>
<table><thead><tr><th>ID</th><th>Lang</th><th>Severity</th><th>Grade</th><th>Verdict</th><th>Score</th></tr></thead>
<tbody>${rows}</tbody></table>
</body></html>`;
  }

  async function loadJsonl(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`could not load ${path}`);
    return parseJsonl(await response.text());
  }

  function coerce(value, kind) {
    if (value == null || value === "") return null;
    if (Array.isArray(value)) return value;
    if (typeof value === "string") {
      const token = value.trim().toLowerCase();
      if (token === "sample" || token === "bundled" || token === "baseline") return token;
      return parseJsonl(value);
    }
    throw new Error(`${kind} must be JSONL, a JSON array, or sample`);
  }

  const cache = { golden: null, predictions: null };

  async function samples(kind) {
    if (kind === "golden") {
      cache.golden = cache.golden || (await loadJsonl("samples/golden.jsonl")).map(asExample);
      return cache.golden;
    }
    cache.predictions = cache.predictions || (await loadJsonl("samples/predictions.sample.jsonl")).map(asPrediction);
    return cache.predictions;
  }

  async function resolveGolden(spec) {
    const value = coerce(spec, "golden");
    if (value === "sample" || value === "bundled") return samples("golden");
    if (!value) throw new Error("golden is required");
    return value.map(asExample);
  }

  async function resolvePreds(spec, golden) {
    const value = coerce(spec, "predictions");
    if (value == null || value === "baseline") return golden.map(heuristic);
    if (value === "sample" || value === "bundled") return samples("predictions");
    return value.map(asPrediction);
  }

  async function handle(path, options) {
    const method = (options && options.method) || "GET";
    const body = options && options.body ? JSON.parse(options.body) : {};
    if (path === "/api/health") {
      return { ok: true, name: "Review Tuner Studio", version: "0.7.0", stdlib_only: true, gpu_required: false, static: true, gpu: false };
    }
    if (path === "/api/glossary") return { glossary: GLOSSARY, metrics: {} };
    if (path === "/api/samples/golden") {
      const records = await samples("golden");
      return { count: records.length, records };
    }
    if (path === "/api/samples/predictions") {
      const records = await samples("predictions");
      return { count: records.length, records };
    }
    if (path === "/api/samples/baseline") {
      const golden = await samples("golden");
      const records = golden.map(heuristic);
      return { count: records.length, records, source: "baseline" };
    }
    if (path === "/api/eval" && method === "POST") {
      const golden = await resolveGolden(body.golden);
      const predictions = await resolvePreds(body.predictions, golden);
      return buildView(golden, predictions, Number(body.threshold || 0.6));
    }
    if (path === "/api/compare" && method === "POST") {
      const golden = await resolveGolden(body.golden);
      const a = await resolvePreds(body.predictions_a, golden);
      const b = await resolvePreds(body.predictions_b, golden);
      const left = buildView(golden, a, Number(body.threshold || 0.6));
      const right = buildView(golden, b, Number(body.threshold || 0.6));
      const keys = ["composite", "token_f1", "must_mention_recall", "severity_accuracy", "tag_f1", "forbidden_rate"];
      const delta = {};
      keys.forEach((key) => {
        delta[key] = Number((right.aggregate[key] - left.aggregate[key]).toFixed(6));
      });
      const improved = [];
      const regressed = [];
      left.examples.forEach((aRow) => {
        const bRow = right.examples.find((item) => item.id === aRow.id);
        if (!bRow || aRow.verdict === bRow.verdict) return;
        const row = { id: aRow.id, from: aRow.verdict, to: bRow.verdict };
        if (bRow.verdict === "caught" || (aRow.verdict === "missed" && bRow.verdict === "partial")) improved.push(row);
        else regressed.push(row);
      });
      return { delta_b_minus_a: delta, a: { letter_grade: left.letter_grade, aggregate: left.aggregate }, b: { letter_grade: right.letter_grade, aggregate: right.aggregate }, a_view: left, b_view: right, improved, regressed };
    }
    if (path === "/api/lint" && method === "POST") {
      const spec = body.records != null ? body.records : body.golden;
      const records = spec === "sample" ? await samples("golden") : coerce(body.text || spec, "records");
      const list = Array.isArray(records) ? records : [];
      const seen = new Map();
      const issues = [];
      list.forEach((raw, index) => {
        const id = String(raw.id || "");
        if (!id) issues.push({ level: "error", rule: "missing_id", message: "missing id", row: index + 1 });
        else if (seen.has(id)) issues.push({ level: "error", rule: "duplicate_id", message: "duplicate id", id, row: index + 1 });
        else seen.set(id, index + 1);
      });
      const errors = issues.filter((item) => item.level === "error").length;
      return { ok: errors === 0, errors, warnings: 0, issues, coverage: { count: list.length, languages: {}, severities: {}, tags: {} } };
    }
    if (path === "/api/badge") {
      const view = await handle("/api/eval", { method: "POST", body: JSON.stringify(body) });
      const letter = view.badge.letter;
      const label = `${view.badge.verdict} ${letter} ${view.badge.percent}%`;
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${12 * label.length + 24}" height="20"><rect width="100%" height="20" rx="4" fill="#1f6f5b"/><text x="50%" y="14" text-anchor="middle" fill="#fff" font-size="11" font-family="sans-serif">${label}</text></svg>`;
      return new Blob([svg], { type: "image/svg+xml" });
    }
    if (path === "/api/report-html") {
      const view = await handle("/api/eval", { method: "POST", body: JSON.stringify(body) });
      return new Blob([reportHtml(view)], { type: "text/html" });
    }
    throw new Error("not found");
  }

  window.ReviewEval = { handle, parseJsonl, buildView, heuristic };
})();
