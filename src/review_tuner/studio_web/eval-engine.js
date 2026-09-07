/* global module */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.ReviewEvalEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PUNCTUATION = new Set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".split(""));
  const SEVERITY_ORDER = ["blocker", "high", "medium", "low", "nit"];
  const VALID_SEVERITIES = new Set(SEVERITY_ORDER);
  const HIGH_IMPACT = new Set(["blocker", "high"]);
  const SCORE_WEIGHTS = Object.freeze({
    tokenF1: 0.35,
    mustMentionRecall: 0.25,
    severityAccuracy: 0.2,
    tagF1: 0.15,
    forbiddenAbsence: 0.05,
  });
  const DEFAULT_THRESHOLD = 0.6;

  function normalizeText(value) {
    const chars = [];
    const lower = String(value).toLowerCase();
    for (let i = 0; i < lower.length; i += 1) {
      const char = lower[i];
      chars.push(PUNCTUATION.has(char) ? " " : char);
    }
    return chars.join("").split(/\s+/).filter(Boolean).join(" ");
  }

  function tokenize(value) {
    const normalized = normalizeText(value);
    return normalized ? normalized.split(" ") : [];
  }

  function countTokens(tokens) {
    const counts = new Map();
    for (const token of tokens) {
      counts.set(token, (counts.get(token) || 0) + 1);
    }
    return counts;
  }

  function bagOverlap(left, right) {
    let overlap = 0;
    for (const [token, count] of left.entries()) {
      if (right.has(token)) {
        overlap += Math.min(count, right.get(token));
      }
    }
    return overlap;
  }

  function tokenF1FromTokens(predTokens, targetTokens) {
    if (!predTokens.length && !targetTokens.length) return 1;
    if (!predTokens.length || !targetTokens.length) return 0;
    const overlap = bagOverlap(countTokens(predTokens), countTokens(targetTokens));
    if (overlap === 0) return 0;
    const precision = overlap / predTokens.length;
    const recall = overlap / targetTokens.length;
    return (2 * precision * recall) / (precision + recall);
  }

  function tokenF1(prediction, target) {
    return tokenF1FromTokens(tokenize(prediction), tokenize(target));
  }

  function phraseRate(normalizedPrediction, phrases) {
    if (!phrases.length) return 0;
    let hits = 0;
    for (const phrase of phrases) {
      if (normalizedPrediction.includes(normalizeText(phrase))) hits += 1;
    }
    return hits / phrases.length;
  }

  function phraseRecall(prediction, requiredPhrases) {
    if (!requiredPhrases.length) return 1;
    return phraseRate(normalizeText(prediction), requiredPhrases);
  }

  function forbiddenRate(prediction, forbiddenPhrases) {
    if (!forbiddenPhrases.length) return 0;
    return phraseRate(normalizeText(prediction), forbiddenPhrases);
  }

  function missedPhrases(prediction, requiredPhrases) {
    const normalized = normalizeText(prediction);
    return requiredPhrases.filter((phrase) => !normalized.includes(normalizeText(phrase)));
  }

  function hitPhrases(prediction, phrases) {
    const normalized = normalizeText(prediction);
    return phrases.filter((phrase) => normalized.includes(normalizeText(phrase)));
  }

  function f1ForSets(predicted, expected) {
    const predictedSet = new Set(predicted);
    const expectedSet = new Set(expected);
    if (!predictedSet.size && !expectedSet.size) return 1;
    if (!predictedSet.size || !expectedSet.size) return 0;
    let overlap = 0;
    for (const item of predictedSet) {
      if (expectedSet.has(item)) overlap += 1;
    }
    if (overlap === 0) return 0;
    const precision = overlap / predictedSet.size;
    const recall = overlap / expectedSet.size;
    return (2 * precision * recall) / (precision + recall);
  }

  function inferSeverity(text) {
    const normalized = normalizeText(text);
    const match = normalized.match(/\bseverity\s+(blocker|high|medium|low|nit)\b/);
    if (match) return match[1];
    const lower = String(text).toLowerCase();
    for (const severity of SEVERITY_ORDER) {
      if (lower.includes("[" + severity + "]") || normalized.includes(severity + " severity")) {
        return severity;
      }
    }
    return null;
  }

  function ngramCounts(tokens, size) {
    const counts = new Map();
    if (size <= 0 || tokens.length < size) return counts;
    for (let i = 0; i <= tokens.length - size; i += 1) {
      const key = tokens.slice(i, i + size).join("\0");
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return counts;
  }

  function mapOverlap(left, right) {
    let overlap = 0;
    for (const [key, count] of left.entries()) {
      if (right.has(key)) overlap += Math.min(count, right.get(key));
    }
    return overlap;
  }

  function bleuLite(prediction, target) {
    const predTokens = tokenize(prediction);
    const targetTokens = tokenize(target);
    if (!predTokens.length && !targetTokens.length) return 1;
    if (!predTokens.length || !targetTokens.length) return 0;
    const precisions = [];
    for (const size of [1, 2]) {
      const predictedNgrams = ngramCounts(predTokens, size);
      const targetNgrams = ngramCounts(targetTokens, size);
      let predictedCount = 0;
      let targetCount = 0;
      for (const count of predictedNgrams.values()) predictedCount += count;
      for (const count of targetNgrams.values()) targetCount += count;
      if (predictedCount === 0 || targetCount === 0) continue;
      precisions.push(mapOverlap(predictedNgrams, targetNgrams) / predictedCount);
    }
    if (!precisions.length) return 0;
    let geometricMean = 0;
    if (precisions.every((value) => value > 0)) {
      let logSum = 0;
      for (const value of precisions) logSum += Math.log(value);
      geometricMean = Math.exp(logSum / precisions.length);
    }
    const candidateLen = predTokens.length;
    const referenceLen = targetTokens.length;
    const brevityPenalty =
      candidateLen >= referenceLen ? 1 : Math.exp(1 - referenceLen / candidateLen);
    return brevityPenalty * geometricMean;
  }

  function lcsLength(left, right) {
    if (!left.length || !right.length) return 0;
    let a = left;
    let b = right;
    if (b.length > a.length) {
      a = right;
      b = left;
    }
    let previous = new Array(b.length + 1).fill(0);
    for (const token of a) {
      const current = new Array(b.length + 1).fill(0);
      for (let j = 1; j <= b.length; j += 1) {
        if (token === b[j - 1]) current[j] = previous[j - 1] + 1;
        else current[j] = previous[j] >= current[j - 1] ? previous[j] : current[j - 1];
      }
      previous = current;
    }
    return previous[b.length];
  }

  function rougeLLite(prediction, target) {
    const predTokens = tokenize(prediction);
    const targetTokens = tokenize(target);
    if (!predTokens.length && !targetTokens.length) return 1;
    if (!predTokens.length || !targetTokens.length) return 0;
    const lcs = lcsLength(predTokens, targetTokens);
    if (lcs === 0) return 0;
    const precision = lcs / predTokens.length;
    const recall = lcs / targetTokens.length;
    return (2 * precision * recall) / (precision + recall);
  }

  function lengthRatio(prediction, target) {
    const predTokens = tokenize(prediction);
    const targetTokens = tokenize(target);
    if (!predTokens.length && !targetTokens.length) return 1;
    if (!targetTokens.length) return predTokens.length;
    return predTokens.length / targetTokens.length;
  }

  function securityFail(severity, mustMentionRecall) {
    return HIGH_IMPACT.has(severity) && mustMentionRecall < 1 ? 1 : 0;
  }

  function stringList(raw, name) {
    if (raw == null) return [];
    if (!Array.isArray(raw)) throw new Error(name + " must be a list of strings");
    const out = [];
    const seen = new Set();
    raw.forEach((value, index) => {
      if (typeof value !== "string" || !value.trim()) {
        throw new Error(name + "[" + index + "] must be a non-empty string");
      }
      const item = value.trim().toLowerCase();
      if (!seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    });
    return out;
  }

  function requiredString(raw, name) {
    const value = raw[name];
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(name + " must be a non-empty string");
    }
    return value;
  }

  function parseGolden(raw) {
    const reviewComment = raw.review_comment;
    const expectedComment = raw.expected_comment;
    let target;
    if (reviewComment != null && expectedComment != null) {
      if (reviewComment !== expectedComment) {
        throw new Error("review_comment and expected_comment must not conflict");
      }
      target = reviewComment;
    } else {
      target = reviewComment != null ? reviewComment : expectedComment;
    }
    if (typeof target !== "string" || !target.trim()) {
      throw new Error("review_comment or expected_comment must be a non-empty string");
    }
    let severity = raw.severity == null ? "medium" : raw.severity;
    if (typeof severity !== "string") throw new Error("severity must be a string");
    severity = severity.trim().toLowerCase();
    if (!VALID_SEVERITIES.has(severity)) {
      throw new Error("severity must be one of " + SEVERITY_ORDER.join(", "));
    }
    const rubric = raw.rubric == null ? {} : raw.rubric;
    if (typeof rubric !== "object" || Array.isArray(rubric)) {
      throw new Error("rubric must be an object when provided");
    }
    return {
      id: requiredString(raw, "id").trim(),
      diff: requiredString(raw, "diff"),
      file_path: requiredString(raw, "file_path").trim(),
      language: requiredString(raw, "language").trim().toLowerCase(),
      context: requiredString(raw, "context"),
      target_comment: target,
      severity,
      tags: stringList(raw.tags, "tags"),
      must_mention: stringList(rubric.must_mention, "rubric.must_mention"),
      avoid: stringList(rubric.avoid, "rubric.avoid"),
    };
  }

  function parsePrediction(raw) {
    let severity = raw.severity == null ? null : raw.severity;
    if (severity != null) {
      if (typeof severity !== "string") throw new Error("severity must be a string or null");
      severity = severity.trim().toLowerCase();
      if (!VALID_SEVERITIES.has(severity)) {
        throw new Error("severity must be one of " + SEVERITY_ORDER.join(", "));
      }
    }
    return {
      id: requiredString(raw, "id").trim(),
      prediction: requiredString(raw, "prediction"),
      severity,
      tags: stringList(raw.tags, "tags"),
    };
  }

  function parseJsonl(text, parser) {
    const records = [];
    const lines = String(text).split(/\r?\n/);
    lines.forEach((line, index) => {
      const stripped = line.trim();
      if (!stripped) return;
      let raw;
      try {
        raw = JSON.parse(stripped);
      } catch (err) {
        throw new Error("line " + (index + 1) + ": invalid JSON");
      }
      if (typeof raw !== "object" || raw == null || Array.isArray(raw)) {
        throw new Error("line " + (index + 1) + ": expected a JSON object");
      }
      try {
        records.push(parser(raw));
      } catch (err) {
        throw new Error("line " + (index + 1) + ": " + err.message);
      }
    });
    return records;
  }

  function scoreExample(golden, prediction, weights) {
    const w = weights || SCORE_WEIGHTS;
    const normalizedPrediction = normalizeText(prediction.prediction);
    const normalizedTarget = normalizeText(golden.target_comment);
    const exact = normalizedPrediction === normalizedTarget ? 1 : 0;
    const lexical = tokenF1FromTokens(
      normalizedPrediction ? normalizedPrediction.split(" ") : [],
      normalizedTarget ? normalizedTarget.split(" ") : []
    );
    const mention = golden.must_mention.length
      ? phraseRate(normalizedPrediction, golden.must_mention)
      : 1;
    const forbidden = golden.avoid.length ? phraseRate(normalizedPrediction, golden.avoid) : 0;
    const predictedSeverity = prediction.severity || inferSeverity(prediction.prediction);
    const severityAccuracy = predictedSeverity === golden.severity ? 1 : 0;
    const tags = f1ForSets(prediction.tags || [], golden.tags || []);
    const composite =
      w.tokenF1 * lexical +
      w.mustMentionRecall * mention +
      w.severityAccuracy * severityAccuracy +
      w.tagF1 * tags +
      w.forbiddenAbsence * (1 - forbidden);
    return {
      id: golden.id,
      exact_match: exact,
      token_f1: lexical,
      must_mention_recall: mention,
      forbidden_rate: forbidden,
      severity_accuracy: severityAccuracy,
      tag_f1: tags,
      composite,
      bleu_lite: bleuLite(prediction.prediction, golden.target_comment),
      rouge_l_lite: rougeLLite(prediction.prediction, golden.target_comment),
      length_ratio: lengthRatio(prediction.prediction, golden.target_comment),
      security_fail: securityFail(golden.severity, mention),
      missed_must_mention: missedPhrases(prediction.prediction, golden.must_mention || []),
      hit_avoid: hitPhrases(prediction.prediction, golden.avoid || []),
    };
  }

  function heuristicPrediction(example) {
    const diffLower = String(example.diff).toLowerCase();
    let comment;
    let tags;
    if (
      diffLower.includes("password") ||
      diffLower.includes("token") ||
      diffLower.includes("auth")
    ) {
      comment =
        "This looks security-sensitive. Please keep the existing validation path and add " +
        "a regression test so expired or invalid tokens are rejected before state changes.";
      tags = ["security", "tests"];
    } else if (diffLower.includes("except") || diffLower.includes("catch")) {
      comment =
        "This error handling hides the failure path. Please preserve the exception details " +
        "or surface a clear typed error so callers can recover safely.";
      tags = ["reliability"];
    } else if (diffLower.includes("select *") || diffLower.includes("for row in")) {
      comment =
        "This may do unnecessary work as data grows. Please narrow the query or batch the " +
        "loop and add a test that covers the large input case.";
      tags = ["performance", "tests"];
    } else {
      comment =
        "Please add a focused regression test for this behavior and make the failure mode " +
        "explicit for future maintainers.";
      tags = ["tests"];
    }
    return {
      id: example.id,
      prediction: comment,
      severity: example.severity,
      tags,
    };
  }

  function mean(values) {
    if (!values.length) return 0;
    let total = 0;
    for (const value of values) total += value;
    return total / values.length;
  }

  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s += 0x6d2b79f5;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function bootstrapCi(values, samples, seed) {
    if (!values || values.length < 3) return null;
    const n = values.length;
    const rounds = samples == null ? 1000 : samples;
    const rng = mulberry32(seed == null ? 1337 : seed);
    const means = [];
    for (let i = 0; i < rounds; i += 1) {
      let total = 0;
      for (let j = 0; j < n; j += 1) {
        total += values[Math.floor(rng() * n)];
      }
      means.push(total / n);
    }
    means.sort((a, b) => a - b);
    return {
      low: means[Math.floor(0.025 * rounds)],
      high: means[Math.min(rounds - 1, Math.floor(0.975 * rounds))],
      samples: rounds,
      seed: seed == null ? 1337 : seed,
    };
  }

  function severityConfusion(examples, predictions) {
    const matrix = {};
    for (const expected of SEVERITY_ORDER) {
      matrix[expected] = {};
      for (const predicted of SEVERITY_ORDER) matrix[expected][predicted] = 0;
      matrix[expected].unknown = 0;
    }
    examples.forEach((example, index) => {
      const predicted = predictions[index] && predictions[index].severity
        ? predictions[index].severity
        : "unknown";
      if (!matrix[example.severity]) return;
      if (matrix[example.severity][predicted] == null) matrix[example.severity].unknown += 1;
      else matrix[example.severity][predicted] += 1;
    });
    return matrix;
  }

  function stddev(values) {
    if (values.length <= 1) return 0;
    const avg = mean(values);
    let sum = 0;
    for (const value of values) {
      const delta = value - avg;
      sum += delta * delta;
    }
    return Math.sqrt(sum / values.length);
  }

  function letterGrade(composite) {
    if (composite >= 0.9) return "A";
    if (composite >= 0.8) return "B";
    if (composite >= 0.7) return "C";
    if (composite >= 0.6) return "D";
    return "F";
  }

  function gateStatus(composite, threshold) {
    const cutoff = threshold == null ? DEFAULT_THRESHOLD : threshold;
    if (composite >= cutoff) return "pass";
    if (composite >= cutoff - 0.1) return "weak";
    return "fail";
  }

  function sliceComposites(examples, scores) {
    const byLanguage = new Map();
    const bySeverity = new Map();
    const byTag = new Map();
    function push(map, key, value) {
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(value);
    }
    examples.forEach((example, index) => {
      const composite = scores[index].composite;
      push(byLanguage, example.language, composite);
      push(bySeverity, example.severity, composite);
      for (const tag of example.tags) push(byTag, tag, composite);
    });
    function freeze(map) {
      const out = {};
      const keys = Array.from(map.keys()).sort();
      for (const key of keys) out[key] = mean(map.get(key));
      return out;
    }
    return {
      by_language: freeze(byLanguage),
      by_severity: freeze(bySeverity),
      by_tag: freeze(byTag),
    };
  }

  function englishSummary(aggregate, examples, scores) {
    const parts = [
      "Grade " +
        aggregate.letter_grade +
        " with mean composite " +
        aggregate.composite.toFixed(3) +
        " on " +
        aggregate.count +
        " examples (" +
        aggregate.gate +
        " vs threshold " +
        Number(aggregate.threshold).toFixed(2) +
        ").",
    ];
    if (aggregate.security_fail_rate > 0) {
      parts.push(
        "Security-fail rate is " +
          Math.round(aggregate.security_fail_rate * 100) +
          "%: at least one blocker/high case missed a required phrase."
      );
    } else {
      parts.push("All blocker/high cases mentioned every required phrase.");
    }
    if (examples.length && scores.length) {
      const ranked = examples
        .map((example, index) => ({ example, score: scores[index] }))
        .sort((a, b) => a.score.composite - b.score.composite)
        .slice(0, 3);
      parts.push(
        "Weakest examples: " +
          ranked
            .map(
              (row) =>
                row.example.id +
                " (" +
                row.example.language +
                "/" +
                row.example.severity +
                ", " +
                row.score.composite.toFixed(3) +
                ")"
            )
            .join(", ") +
          "."
      );
    }
    const byLanguage = aggregate.slices && aggregate.slices.by_language;
    if (byLanguage && Object.keys(byLanguage).length) {
      let worstKey = null;
      let worstValue = Infinity;
      for (const [key, value] of Object.entries(byLanguage)) {
        if (value < worstValue) {
          worstKey = key;
          worstValue = value;
        }
      }
      if (worstKey != null) {
        parts.push("Lowest language slice is " + worstKey + " at " + worstValue.toFixed(3) + ".");
      }
    }
    return parts.join(" ");
  }

  function evaluateDataset(golden, predictions, threshold) {
    const cutoff = threshold == null ? DEFAULT_THRESHOLD : threshold;
    const predictionById = new Map();
    for (const prediction of predictions) {
      if (predictionById.has(prediction.id)) {
        throw new Error("duplicate id: " + prediction.id);
      }
      predictionById.set(prediction.id, prediction);
    }
    const goldenIds = new Set();
    const missing = [];
    const scores = [];
    const alignedGolden = [];
    const alignedPredictions = [];
    for (const example of golden) {
      if (goldenIds.has(example.id)) throw new Error("duplicate golden id: " + example.id);
      goldenIds.add(example.id);
      const prediction = predictionById.get(example.id);
      if (!prediction) {
        missing.push(example.id);
        continue;
      }
      alignedGolden.push(example);
      alignedPredictions.push(prediction);
      scores.push(scoreExample(example, prediction));
    }
    if (!goldenIds.size) throw new Error("golden dataset must contain at least one example");
    const extra = Array.from(predictionById.keys())
      .filter((id) => !goldenIds.has(id))
      .sort();
    const alignment = [];
    if (missing.length) alignment.push("missing predictions for ids: " + missing.join(", "));
    if (extra.length) alignment.push("unexpected predictions for ids: " + extra.join(", "));
    if (alignment.length) throw new Error(alignment.join("; "));

    function collect(field) {
      return scores.map((score) => score[field]);
    }
    const compositeValues = collect("composite");
    const aggregate = {
      count: scores.length,
      exact_match: mean(collect("exact_match")),
      exact_match_std: stddev(collect("exact_match")),
      token_f1: mean(collect("token_f1")),
      token_f1_std: stddev(collect("token_f1")),
      must_mention_recall: mean(collect("must_mention_recall")),
      must_mention_recall_std: stddev(collect("must_mention_recall")),
      forbidden_rate: mean(collect("forbidden_rate")),
      forbidden_rate_std: stddev(collect("forbidden_rate")),
      severity_accuracy: mean(collect("severity_accuracy")),
      severity_accuracy_std: stddev(collect("severity_accuracy")),
      tag_f1: mean(collect("tag_f1")),
      tag_f1_std: stddev(collect("tag_f1")),
      composite: mean(compositeValues),
      composite_std: stddev(compositeValues),
      bleu_lite: mean(collect("bleu_lite")),
      rouge_l_lite: mean(collect("rouge_l_lite")),
      length_ratio: mean(collect("length_ratio")),
      security_fail_rate: mean(collect("security_fail")),
    };
    aggregate.composite_percent = Math.floor(aggregate.composite * 10000) / 100;
    aggregate.slices = sliceComposites(alignedGolden, scores);
    aggregate.letter_grade = letterGrade(aggregate.composite);
    aggregate.gate = gateStatus(aggregate.composite, cutoff);
    aggregate.threshold = cutoff;
    aggregate.bootstrap_ci = bootstrapCi(compositeValues, 1000, 1337);
    aggregate.severity_confusion = severityConfusion(alignedGolden, alignedPredictions);
    aggregate.summary = englishSummary(aggregate, alignedGolden, scores);
    if (aggregate.bootstrap_ci) {
      aggregate.summary +=
        " Browser bootstrap 95% CI on composite: " +
        aggregate.bootstrap_ci.low.toFixed(3) +
        " – " +
        aggregate.bootstrap_ci.high.toFixed(3) +
        ".";
    }
    const details = alignedGolden.map((example, index) => {
      const prediction = alignedPredictions[index];
      const score = scores[index];
      return Object.assign({}, score, {
        language: example.language,
        file_path: example.file_path,
        severity_expected: example.severity,
        tags: example.tags.slice(),
        diff: example.diff,
        expected_comment: example.target_comment,
        prediction: prediction.prediction,
        predicted_severity: prediction.severity,
        predicted_tags: prediction.tags.slice(),
      });
    });
    return { aggregate, scores, details, golden: alignedGolden, predictions: alignedPredictions };
  }

  function compareDatasets(golden, predictionsA, predictionsB, threshold) {
    const a = evaluateDataset(golden, predictionsA, threshold);
    const b = evaluateDataset(golden, predictionsB, threshold);
    const newlyCaught = [];
    const newlyMissed = [];
    const perExample = a.scores.map((scoreA, index) => {
      const scoreB = b.scores[index];
      const caughtA = scoreA.must_mention_recall >= 1;
      const caughtB = scoreB.must_mention_recall >= 1;
      if (caughtB && !caughtA) newlyCaught.push(scoreA.id);
      if (caughtA && !caughtB) newlyMissed.push(scoreA.id);
      return {
        id: scoreA.id,
        composite_a: scoreA.composite,
        composite_b: scoreB.composite,
        delta_composite: scoreB.composite - scoreA.composite,
        caught_a: caughtA,
        caught_b: caughtB,
      };
    });
    return {
      a,
      b,
      delta_composite: b.aggregate.composite - a.aggregate.composite,
      newly_caught: newlyCaught,
      newly_missed: newlyMissed,
      per_example: perExample,
    };
  }

  return {
    SCORE_WEIGHTS,
    DEFAULT_THRESHOLD,
    SEVERITY_ORDER,
    normalizeText,
    tokenize,
    tokenF1,
    phraseRecall,
    forbiddenRate,
    missedPhrases,
    f1ForSets,
    inferSeverity,
    bleuLite,
    rougeLLite,
    lengthRatio,
    securityFail,
    scoreExample,
    heuristicPrediction,
    parseGolden,
    parsePrediction,
    parseJsonl,
    letterGrade,
    gateStatus,
    sliceComposites,
    bootstrapCi,
    severityConfusion,
    evaluateDataset,
    compareDatasets,
    englishSummary,
  };
});
