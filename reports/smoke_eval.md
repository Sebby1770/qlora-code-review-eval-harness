# Code review eval report

**Examples:** 15

**Composite 95% CI:** 0.4621 – 0.6312

## Metrics

| Metric | Mean | Std |
| --- | ---: | ---: |
| `composite` | 0.5438 | 0.1747 |
| `token_f1` | 0.4166 | 0.1951 |
| `bleu_lite` | 0.1898 | 0.1606 |
| `rouge_l` | 0.3521 | 0.2026 |
| `length_ratio` | 0.6823 | 0.1163 |
| `must_mention_recall` | 0.3778 | 0.4366 |
| `forbidden_rate` | 0.0000 | 0.0000 |
| `severity_accuracy` | 1.0000 | 0.0000 |
| `tag_f1` | 0.7578 | 0.1778 |
| `exact_match` | 0.0000 | 0.0000 |

## By language

- **go**: 0.5126
- **java**: 0.3342
- **javascript**: 0.5744
- **python**: 0.6344
- **ruby**: 0.4313
- **rust**: 0.4005
- **typescript**: 0.3763

## By severity

- **blocker**: 0.5794
- **high**: 0.4973
- **low**: 0.7524
- **medium**: 0.5163

## Error analysis

### Most-missed must_mention

| Phrase | Count |
| --- | ---: |
| expired refresh tokens | 1 |
| expires_at check | 1 |
| limit | 1 |
| full order history | 1 |
| pagination | 1 |
| one extra attempt | 1 |
| retry budget | 1 |
| tests | 1 |
| raw tokens | 1 |
| hashing | 1 |
| constant-time comparison | 1 |
| validation | 1 |
| persistence | 1 |
| unknown fields | 1 |
| tenant | 1 |
| cache key | 1 |
| cross-tenant | 1 |
| panic | 1 |
| result | 1 |
| malformed input | 1 |
| idempotency | 1 |
| double-refund | 1 |
| retries | 1 |
| memory | 1 |
| streaming | 1 |
| scale | 1 |
| sql injection | 1 |
| 24 hours | 1 |

### Forbidden-phrase hits

No forbidden-phrase hits.

### Severity confusion

| Predicted | Gold | Count |
| --- | --- | ---: |
| medium | medium | 6 |
| blocker | blocker | 4 |
| high | high | 4 |
| low | low | 1 |

## Examples (worst composite first)

| ID | Language | Severity | Composite | Mention recall |
| --- | --- | --- | ---: | ---: |
| `golden-008` | java | high | 0.3342 | 0.0000 |
| `golden-003` | go | medium | 0.3468 | 0.0000 |
| `golden-002` | typescript | medium | 0.3763 | 0.0000 |
| `golden-006` | javascript | blocker | 0.3963 | 0.0000 |
| `golden-007` | rust | medium | 0.4005 | 0.0000 |
| `golden-009` | python | medium | 0.4091 | 0.0000 |
| `golden-005` | python | high | 0.4284 | 0.0000 |
| `golden-004` | ruby | blocker | 0.4313 | 0.0000 |
| `golden-001` | python | high | 0.5480 | 0.3333 |
| `golden-014` | go | high | 0.6784 | 1.0000 |
| `golden-012` | python | blocker | 0.7072 | 1.0000 |
| `golden-015` | javascript | low | 0.7524 | 1.0000 |
| `golden-013` | python | medium | 0.7596 | 0.6667 |
| `golden-010` | python | blocker | 0.7828 | 0.6667 |
| `golden-011` | python | medium | 0.8055 | 1.0000 |
