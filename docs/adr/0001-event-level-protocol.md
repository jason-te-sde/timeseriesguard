# 1. Report event-level AUC-PR as the headline

Status: accepted, 2026-11-03, before any model was run on SMD.

## Context

The project's claim is a single AUC-PR on SMD, and the protocol that turns per-row scores into that
number decides what it means. SMD's 327 test anomaly segments are very uneven in length (median 11
rows, 10th percentile 3, 90th percentile 239, longest 3,161), and the share of anomalous rows per
machine ranges from 0.42 % to 15.65 %.

Most published SMD numbers use point adjust (PA), which credits a whole segment when any of its rows
crosses the threshold. With segments that long, a detector that knows nothing does very well under
it. One seed was measured before the decision was taken: uniform random scores reached a PA F1\* of
0.963 on machine-1-1 and about 0.78 averaged over the 28 machines. The test that now pins the result
measured five seeds later: 0.963 to 0.989 and 0.767 to 0.776.

The protocol had to be fixed before any model existed, so that it could not drift towards whatever a
model happened to be good at.

## Decision

The headline is the **event-level AUC-PR**, averaged over the 28 machines, with constants fixed in
`ProtocolParams`:

- every anomaly segment is one positive sample, scored by its maximum;
- the normal rows, minus a **cooldown of `C = 60`** rows after each segment, are cut into **chunks of
  `L = 60`** rows, each one negative sample scored by its maximum.

One minute per row makes a chunk one hour: an hour with any alert is one false page, the usual
deduplication window for alerts. An alert in the hour after an incident belongs to the same incident.

Point-wise, PA and PA%K (Kim et al., AAAI 2022, §4.2, `K = 20`, a segment counts when *more than*
`K` % of it is detected, implemented as `floor(K * len / 100) + 1` rows) are reported next to it for
every model and machine, with F1 at the frozen threshold and the oracle F1\* marked as an upper bound.
A sensitivity grid recomputes the event-level number for `(L, C)` in `{30, 60, 120} x {0, 60}`.
The full definitions are in [evaluation.md](../evaluation.md).

## Consequences

- Random scores cannot pass for a detector: `tests/protocol` asserts their macro event-level AUC-PR
  stays below 0.1 (measured 0.039 to 0.049) on every one of five seeds, next to the PA result.
- The protocol counts incidents and quiet hours, which is how alerts are experienced on call.

Costs:

- **The headline is not comparable with most published SMD numbers**, and it will look low next to
  them. Reporting PA in the same table is the mitigation; a reader still has to read the protocol.
- **Detection delay is invisible.** A segment detected in its last minute counts as fully as one
  detected in its first.
- **Every segment weighs the same.** A 3-row blip and a 3,161-row incident are one positive each.
- **Alarms in the cooldown are forgiven**, including genuine false alarms right after an incident.
- **Machines with few segments give coarse values.** machine-2-8 has one segment, so its event-level
  AUC-PR can only be 1, 1/2, 1/3, ...; per-machine values and a bootstrap interval over machines are
  reported for that reason.
- `L` and `C` are operational judgements, not measurements. The sensitivity grid shows how much the
  number moves with them; it does not make them right.

## Rejected alternatives

**Point adjust as the headline.** It is what the literature reports, and random scores score as state
of the art under it. A number that a detector which knows nothing can reach says nothing about a
detector.

**Point-wise metrics only.** Point-wise cannot be gamed this way, but it counts rows: the longest
segments dominate it, and it charges a detector for every minute of an incident it did not flag even
after it raised the alarm. On machine-1-1 the best point-wise F1 of random scores is within 0.002 of
flagging every row, which is the floor any detector is compared against, not a useful scale.

**PA%K as the headline.** It repairs the worst of PA, but it still counts rows, the choice of `K` is
arbitrary, and it is defined for one threshold, so average precision needs the score rewriting
described in evaluation.md. It is reported, not headlined.
