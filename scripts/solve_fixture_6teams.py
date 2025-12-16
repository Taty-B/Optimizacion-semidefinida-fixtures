#!/usr/bin/env python
import numpy as np

from src.tournament import Tournament, count_breaks, display_assignment, display_tournament_chart
from src.sdp_solver import solve_break_minimization


def build_fixture_tournament() -> Tournament:
    chart = np.array(
        [
            [5, 2, 4, 1, 3],
            [4, 5, 3, 0, 2],
            [3, 0, 5, 4, 1],
            [2, 4, 1, 5, 0],
            [1, 3, 0, 2, 5],
            [0, 1, 2, 3, 4],
        ],
        dtype=int,
    )
    t = Tournament(n_teams=6, chart=chart)
    assert t.validate(), "Fixture chart is not a valid round-robin."
    return t


def main():
    tournament = build_fixture_tournament()

    print("Tabla del torneo:")
    print(display_tournament_chart(tournament))
    sdp_result, best_result, _ = solve_break_minimization(
        tournament, n_rounds=300, solver="SCS", seed=123
    )

    ha = best_result.assignment.home_away
    total_breaks, hh, aa = count_breaks(tournament, best_result.assignment)

    print("\nMejor asignación (H=1, A=0):")
    print(display_assignment(tournament, best_result.assignment))
    print(f"\nBreaks totales: {total_breaks} (HH: {hh}, AA: {aa})")
    print(f"Non-breaks: {best_result.non_breaks}")
    print(f"Valor SDP (cota superior de non-breaks): {sdp_result.sdp_value:.4f}")

    print("\nAristas consecutivas (por equipo): cut = cambio (non-break), not-cut = break")
    for i in range(tournament.n_teams):
        changes = []
        for s in range(1, tournament.n_slots):
            cut = ha[i, s] != ha[i, s - 1]
            changes.append("cut" if cut else "not-cut")
        print(f"Equipo {i+1}: {changes}")


if __name__ == "__main__":
    main()
