import numpy as np
import cvxpy as cp
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time

from tournament import (
    Tournament, Assignment, count_breaks, count_non_breaks,
    get_games_list, get_consecutive_pairs, generate_circle_method_tournament
)


@dataclass
class SDPResult:
    sdp_value: float
    gram_matrix: np.ndarray
    vectors: np.ndarray
    solve_time: float


@dataclass
class RoundingResult:
    assignment: Assignment
    breaks: int
    non_breaks: int
    objective_ratio: float


def build_sdp_model(tournament: Tournament) -> Tuple[cp.Problem, cp.Variable, Dict]:
    games = get_games_list(tournament)
    n_games = len(games)
    n = tournament.n_teams
    n_slots = tournament.n_slots
    
    cell_to_game = {}
    for g_idx, (i, j, s) in enumerate(games):
        cell_to_game[(i, s)] = (g_idx, 1)
        cell_to_game[(j, s)] = (g_idx, -1)
    
    pairs = get_consecutive_pairs(tournament)
    n_pairs = len(pairs)
    
    dim = n_games + 1
    Y = cp.Variable((dim, dim), symmetric=True)
    
    constraints = []
    
    constraints.append(Y >> 0)
    
    for i in range(dim):
        constraints.append(Y[i, i] == 1)
    
    objective_expr = 0
    
    for team, s_prev, s_curr in pairs:
        g1, sign1 = cell_to_game[(team, s_prev)]
        g2, sign2 = cell_to_game[(team, s_curr)]
        
        y1 = Y[0, g1 + 1]
        y2 = Y[0, g2 + 1]
        y1y2 = Y[g1 + 1, g2 + 1]
        contribution = (3 + sign1 * y1 + sign2 * y2 - sign1 * sign2 * y1y2) / 4
        objective_expr += contribution
    
    objective = cp.Maximize(objective_expr)
    problem = cp.Problem(objective, constraints)
    
    metadata = {
        'games': games,
        'cell_to_game': cell_to_game,
        'pairs': pairs,
        'n_games': n_games,
        'dim': dim
    }
    
    return problem, Y, metadata


def solve_sdp(tournament: Tournament, solver: str = 'SCS', 
              verbose: bool = False) -> SDPResult:
    problem, Y, metadata = build_sdp_model(tournament)
    
    start_time = time.time()
    
    if solver == 'SCS':
        problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)
    elif solver == 'CLARABEL':
        problem.solve(solver=cp.CLARABEL, verbose=verbose)
    else:
        problem.solve(verbose=verbose)
    
    solve_time = time.time() - start_time
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError(f"SDP solver failed with status: {problem.status}")
    
    gram_matrix = Y.value
    
    eigvals, eigvecs = np.linalg.eigh(gram_matrix)
    
    eigvals = np.maximum(eigvals, 0)
    
    sqrt_eigvals = np.sqrt(eigvals)
    vectors = np.diag(sqrt_eigvals) @ eigvecs.T
    
    return SDPResult(
        sdp_value=problem.value,
        gram_matrix=gram_matrix,
        vectors=vectors,
        solve_time=solve_time
    )


def goemans_williamson_rounding(tournament: Tournament, 
                                 sdp_result: SDPResult,
                                 n_rounds: int = 100,
                                 seed: Optional[int] = None) -> List[RoundingResult]:
    rng = np.random.default_rng(seed)
    games = get_games_list(tournament)
    n_games = len(games)
    n = tournament.n_teams
    n_slots = tournament.n_slots
    
    vectors = sdp_result.vectors
    dim = vectors.shape[0]
    
    results = []
    
    for _ in range(n_rounds):
        r = rng.standard_normal(dim)
        r = r / np.linalg.norm(r)
        
        y_values = np.sign(vectors.T @ r)
        
        if y_values[0] < 0:
            y_values = -y_values
        
        game_assignments = y_values[1:]
        
        game_assignments[game_assignments == 0] = 1
        
        home_away = np.zeros((n, n_slots), dtype=int)
        
        for g_idx, (i, j, s) in enumerate(games):
            if game_assignments[g_idx] > 0:
                home_away[i, s] = 1
                home_away[j, s] = 0
            else:
                home_away[i, s] = 0
                home_away[j, s] = 1
        
        assignment = Assignment(home_away=home_away)
        breaks, _, _ = count_breaks(tournament, assignment)
        non_breaks = count_non_breaks(tournament, assignment)
        
        ratio = non_breaks / sdp_result.sdp_value if sdp_result.sdp_value > 0 else 0
        
        results.append(RoundingResult(
            assignment=assignment,
            breaks=breaks,
            non_breaks=non_breaks,
            objective_ratio=ratio
        ))
    
    return results


def solve_break_minimization(tournament: Tournament,
                              n_rounds: int = 100,
                              solver: str = 'SCS',
                              verbose: bool = False,
                              seed: Optional[int] = None) -> Tuple[SDPResult, RoundingResult, List[RoundingResult]]:
    sdp_result = solve_sdp(tournament, solver=solver, verbose=verbose)
    
    rounding_results = goemans_williamson_rounding(
        tournament, sdp_result, n_rounds=n_rounds, seed=seed
    )
    
    best_result = min(rounding_results, key=lambda r: r.breaks)
    
    return sdp_result, best_result, rounding_results


if __name__ == "__main__":
    print("Probando solver SDP para minimización de breaks")
    print("=" * 60)
    
    for n_teams in [4, 6, 8]:
        print(f"\n--- {n_teams} equipos ---")
        
        tournament = generate_circle_method_tournament(n_teams, shuffle=True, seed=42)
        games = get_games_list(tournament)
        pairs = get_consecutive_pairs(tournament)
        
        print(f"Número de partidos: {len(games)}")
        print(f"Número de pares consecutivos (aristas E1): {len(pairs)}")
        print(f"Cota inferior de breaks: {n_teams - 2}")
        
        sdp_result, best_result, all_results = solve_break_minimization(
            tournament, n_rounds=200, solver='SCS', verbose=False, seed=123
        )
        
        print(f"\nValor óptimo SDP (cota superior de non-breaks): {sdp_result.sdp_value:.4f}")
        print(f"Tiempo de resolución SDP: {sdp_result.solve_time:.4f} segundos")
        
        print("\nMejor solución (redondeo):")
        print(f"  Breaks: {best_result.breaks}")
        print(f"  Non-breaks: {best_result.non_breaks}")
        print(f"  Ratio de aproximación: {best_result.objective_ratio:.4f}")
        
        all_non_breaks = [r.non_breaks for r in all_results]
        all_ratios = [r.objective_ratio for r in all_results]
        
        print(f"\nEstadísticas sobre {len(all_results)} rondas:")
        print(f"  Non-breaks promedio: {np.mean(all_non_breaks):.2f}")
        print(f"  Mejor non-breaks: {max(all_non_breaks)}")
        print(f"  Ratio promedio: {np.mean(all_ratios):.4f}")
        print("  Garantía teórica: 0.87856")

