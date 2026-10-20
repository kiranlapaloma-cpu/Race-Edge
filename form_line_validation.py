"""Race Edge Horse Intelligence: read-only subsequent form-line validation.

Pure analysis: accepts the output of database_readonly.historical_race_explorer.
No database writes, external packages, prediction calculations or mean MR values.

Terminology:
- 'placing' is a podium finish (2nd/3rd) by default, NOT a Tote payout.
- MR confirmation tolerance is configurable and provisional until calibrated.
- Only saved subsequent races are considered; absence is not negative evidence.
"""
from __future__ import annotations

from datetime import date
from statistics import median
from typing import Any, Mapping
import re


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if result == result and abs(result) != float('inf') else None
    except (ValueError, TypeError):
        return None


def _finish(value: Any) -> tuple[int | None, int | None]:
    """Parse '2/11' or '2'; invalid positions are excluded from results counts."""
    match = re.fullmatch(r'\s*(\d+)\s*(?:/\s*(\d+)\s*)?', str(value or ''))
    if not match:
        return None, None
    pos = int(match.group(1))
    field_size = int(match.group(2)) if match.group(2) else None
    if pos < 1 or (field_size is not None and (field_size < 1 or pos > field_size)):
        return None, field_size
    return pos, field_size


def _identity(run: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        race_no = int(run.get('race_number') or 0)
    except (ValueError, TypeError):
        race_no = 0
    return (str(run.get('race_date') or ''), str(run.get('track') or '').casefold(),
            str(run.get('course') or '').casefold(), race_no)


def _unique_runs(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for run in sorted(runs, key=lambda r: _identity(r)):
        key = str(run.get('id')) if run.get('id') is not None else _identity(run)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(run))
    return output


def validate_form_line(
    explorer: Mapping[str, Any],
    *,
    follow_up_starts: int = 3,
    mr_tolerance: float = 3.0,
    placing_positions: tuple[int, ...] = (2, 3),
    minimum_confirmations: int = 3,
) -> dict[str, Any]:
    """Validate a historical race from its saved subsequent runners.

    Comparisons are to EACH horse's original MR Achieved, never to field means.
    An MR is 'confirmed' when follow-up MR >= original MR - tolerance.
    A horse is confirmed if ANY of its considered follow-up runs confirms.
    A horse is below only when ALL eligible follow-up MRs are below tolerance.
    'next_start' counts only the first saved subsequent start; 'window' counts
    up to follow_up_starts saved subsequent starts per runner.
    The verdict is deliberately conservative and never a new numeric rating.
    """
    if isinstance(follow_up_starts, bool) or not 1 <= follow_up_starts <= 20:
        raise ValueError('follow_up_starts must be between 1 and 20')
    if _number(mr_tolerance) is None or mr_tolerance < 0:
        raise ValueError('mr_tolerance must be a nonnegative finite number')
    if isinstance(minimum_confirmations, bool) or minimum_confirmations < 1:
        raise ValueError('minimum_confirmations must be positive')
    if not placing_positions or any(isinstance(x, bool) or not isinstance(x, int) or x < 2 for x in placing_positions):
        raise ValueError('placing_positions must contain valid positions >= 2')

    race = dict(explorer.get('race') or {})
    original_day = str(race.get('race_date') or '')
    if original_day:
        date.fromisoformat(original_day)
    summaries = []
    next_winners = next_places = window_winners = window_places = 0
    unique_next_winners = set()
    unique_next_places = set()
    unique_window_winners = set()
    unique_window_places = set()
    confirmed = below = mixed = unassessable = 0
    differences = []
    compared_runners = 0
    runners_with_followups = 0
    total_followup_starts = 0
    all_runners = explorer.get('runners') or []

    for index, item in enumerate(all_runners):
        original = dict(item.get('race_run') or {})
        horse = str(original.get('horse') or '').strip()
        # Stable distinct key even if the data contains duplicate/blank names.
        horse_key = (horse.upper(), index)
        original_mr = _number(original.get('mr_achieved'))
        subsequent = [r for r in (item.get('subsequent_runs') or [])
                      if isinstance(r, Mapping) and str(r.get('race_date') or '') > original_day]
        subsequent = _unique_runs(subsequent)[:follow_up_starts]
        if subsequent:
            runners_with_followups += 1
        total_followup_starts += len(subsequent)
        details = []
        eligible_deltas = []
        for j, run in enumerate(subsequent):
            pos, field_size = _finish(run.get('finish_position'))
            win = pos == 1
            place = pos in placing_positions
            if win:
                window_winners += 1
                unique_window_winners.add(horse_key)
                if j == 0:
                    next_winners += 1
                    unique_next_winners.add(horse_key)
            if place:
                window_places += 1
                unique_window_places.add(horse_key)
                if j == 0:
                    next_places += 1
                    unique_next_places.add(horse_key)
            achieved = _number(run.get('mr_achieved'))
            delta = round(achieved - original_mr, 4) if achieved is not None and original_mr is not None else None
            if delta is not None:
                eligible_deltas.append(delta)
                differences.append(delta)
            details.append({
                'race_date': run.get('race_date'), 'track': run.get('track'),
                'course': run.get('course'), 'race_number': run.get('race_number'),
                'distance': run.get('distance'), 'finish_position': run.get('finish_position'),
                'position': pos, 'field_size': field_size,
                'mr_achieved': achieved, 'mr_change': delta,
                'mr_confirmed': delta >= -mr_tolerance if delta is not None else None,
                'won': win, 'placed': place,
                'race_id': run.get('id'),
            })
        if original_mr is None or not eligible_deltas:
            status = 'insufficient_evidence'
            unassessable += 1
        else:
            compared_runners += 1
            if any(d >= -mr_tolerance for d in eligible_deltas):
                status = 'confirmed'
                confirmed += 1
            else:
                status = 'below_original'
                below += 1
            if any(d >= -mr_tolerance for d in eligible_deltas) and any(d < -mr_tolerance for d in eligible_deltas):
                mixed += 1
        summaries.append({
            'horse': horse, 'original_run_id': original.get('id'),
            'original_finish_position': original.get('finish_position'),
            'original_mr_achieved': original_mr,
            'recorded_subsequent_starts': item.get('subsequent_runs_recorded', len(subsequent)),
            'starts_analysed': len(subsequent), 'mr_status': status,
            'best_subsequent_mr': max((d['mr_achieved'] for d in details if d['mr_achieved'] is not None), default=None),
            'next_recorded_start': details[0] if details else None,
            'subsequent_runs': details,
        })

    # No mean MR values. A median of individual changes is descriptive only.
    median_change = median(differences) if differences else None
    if compared_runners < minimum_confirmations:
        verdict = 'insufficient_evidence'
        explanation = 'Too few runners have subsequent comparable MR Achieved records to validate this form line.'
    elif confirmed >= minimum_confirmations and confirmed > below:
        verdict = 'subsequently_supported'
        explanation = 'Multiple runners reproduced or exceeded their original MR Achieved within the configured tolerance.'
    elif below >= minimum_confirmations and below > confirmed:
        verdict = 'questionable'
        explanation = 'Multiple runners fell below their original MR Achieved in all available follow-up starts.'
    else:
        verdict = 'mixed_evidence'
        explanation = 'Subsequent MR Achieved performances give mixed evidence about the original form line.'

    return {
        'race': race,
        'settings': {
            'follow_up_starts': follow_up_starts,
            'mr_confirmation_tolerance_points': mr_tolerance,
            'placing_positions': list(placing_positions),
            'placing_definition': 'Finishing positions, not Tote dividend-paying places',
            'minimum_confirmations': minimum_confirmations,
        },
        'coverage': {
            'saved_original_runners': len(summaries),
            'runners_with_saved_followups': runners_with_followups,
            'runners_with_mr_comparisons': compared_runners,
            'followup_starts_analysed': total_followup_starts,
            'note': 'Only recorded Race Edge starts are included. Missing starts are unknown, not defeats.'
        },
        'mr_validation': {
            'confirmed_runners': confirmed, 'below_original_runners': below,
            'mixed_followup_runners': mixed, 'insufficient_evidence_runners': unassessable,
            'median_individual_mr_change': median_change,
            'note': 'Confirmation is per horse (at least one run within tolerance), not a mean MR.'
        },
        'subsequent_results': {
            'next_recorded_start': {
                'wins': next_winners, 'placings_excluding_wins': next_places,
                'unique_winners': len(unique_next_winners),
                'unique_placed_horses': len(unique_next_places),
            },
            'followup_window': {
                'wins': window_winners, 'placings_excluding_wins': window_places,
                'unique_winners': len(unique_window_winners),
                'unique_placed_horses': len(unique_window_places),
            },
        },
        'verdict': verdict,
        'verdict_explanation': explanation,
        'verdict_caveat': 'A retrospective indicator, not a causal proof of form quality; distance, course, timing and opportunities can differ.',
        'runners': summaries,
    }


def validate_saved_race(day: str, track: str, course: str, race_number: int, *,
                        follow_up_starts: int = 3, mr_tolerance: float = 3.0) -> dict[str, Any]:
    """Convenience entry point; uses existing read-only Historical Race Explorer."""
    from database_readonly import historical_race_explorer
    explorer = historical_race_explorer(day, track, course, race_number,
                                        subsequent_limit=follow_up_starts)
    return validate_form_line(explorer, follow_up_starts=follow_up_starts,
                              mr_tolerance=mr_tolerance)
