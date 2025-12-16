import numpy as np
from typing import Tuple, Optional, List
from itertools import product

from tournament import (
    Tournament, Assignment, count_breaks, count_non_breaks,
    get_games_list, generate_circle_method_tournament
)


def brute_force_optimal(tournament: Tournament) -> Tuple[Assignment, int, int]:
    games = get_games_list(tournament)
    n_games = len(games)
    n = tournament.n_teams
    n_slots = tournament.n_slots
    
    if n_games > 20:
        raise ValueError(f"Too many games ({n_games}) for brute force. Use n_teams <= 6.")
    
    best_assignment = None
    min_breaks = float('inf')
    
    for bits in range(2 ** n_games):
        home_away = np.zeros((n, n_slots), dtype=int)
        
        for g, (i, j, s) in enumerate(games):
            if (bits >> g) & 1 == 0:
                home_away[i, s] = 1
                home_away[j, s] = 0
            else:
                home_away[i, s] = 0
                home_away[j, s] = 1
        
        assignment = Assignment(home_away=home_away)
        breaks, _, _ = count_breaks(tournament, assignment)
        
        if breaks < min_breaks:
            min_breaks = breaks
            best_assignment = assignment
    
    max_non_breaks = count_non_breaks(tournament, best_assignment)
    return best_assignment, min_breaks, max_non_breaks


def backtracking_solver(tournament: Tournament, 
                        early_termination: Optional[int] = None) -> Tuple[Assignment, int, int]:
    games = get_games_list(tournament)
    n_games = len(games)
    n = tournament.n_teams
    n_slots = tournament.n_slots
    
    home_away = np.zeros((n, n_slots), dtype=int)
    assigned = np.zeros((n, n_slots), dtype=bool)
    
    best_breaks = [float('inf')]
    best_assignment = [None]
    
    lower_bound = n - 2
    
    def count_current_breaks() -> int:
        breaks = 0
        for i in range(n):
            for s in range(1, n_slots):
                if assigned[i, s-1] and assigned[i, s]:
                    if home_away[i, s-1] == home_away[i, s]:
                        breaks += 1
        return breaks
    
    def lower_bound_remaining(game_idx: int) -> int:
        return 0
    
    def backtrack(game_idx: int):
        if game_idx == n_games:
            breaks = count_current_breaks()
            if breaks < best_breaks[0]:
                best_breaks[0] = breaks
                best_assignment[0] = Assignment(home_away=home_away.copy())
            return
        
        current_breaks = count_current_breaks()
        if current_breaks >= best_breaks[0]:
            return
        
        if early_termination is not None and best_breaks[0] <= early_termination:
            return
        
        i, j, s = games[game_idx]
        
        for home_team in [i, j]:
            away_team = j if home_team == i else i
            
            home_away[home_team, s] = 1
            home_away[away_team, s] = 0
            assigned[home_team, s] = True
            assigned[away_team, s] = True
            
            backtrack(game_idx + 1)
            
            assigned[home_team, s] = False
            assigned[away_team, s] = False
    
    backtrack(0)
    
    max_non_breaks = count_non_breaks(tournament, best_assignment[0])
    return best_assignment[0], best_breaks[0], max_non_breaks


def verify_solution(tournament: Tournament, assignment: Assignment) -> bool:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    ha = assignment.home_away
    
    for s in range(n_slots):
        homes_in_slot = 0
        for i in range(n):
            j = tournament.chart[i, s]
            if i < j:
                if ha[i, s] + ha[j, s] != 1:
                    print(f"Asignación inválida del partido: equipos {i+1} y {j+1} en la fecha {s+1}")
                    return False
            homes_in_slot += ha[i, s]
        
        if homes_in_slot != n // 2:
            print(f"Fecha desbalanceada {s+1}: {homes_in_slot} locales en vez de {n//2}")
            return False
    
    return True


if __name__ == "__main__":
    print("Probando solver exacto")
    print("=" * 50)
    
    print("\n--- 4 equipos ---")
    tournament4 = generate_circle_method_tournament(4, shuffle=False, seed=42)
    print(f"Número de partidos: {len(get_games_list(tournament4))}")
    
    best_assign, min_breaks, max_non_breaks = brute_force_optimal(tournament4)
    print(f"Breaks óptimos (fuerza bruta): {min_breaks}")
    print(f"Máx non-breaks: {max_non_breaks}")
    print(f"Cota inferior: {4 - 2} = 2")
    print(f"Solución válida: {verify_solution(tournament4, best_assign)}")
    
    best_assign_bt, min_breaks_bt, _ = backtracking_solver(tournament4)
    print(f"Breaks óptimos (backtracking): {min_breaks_bt}")
    assert min_breaks == min_breaks_bt, "Mismatch between brute force and backtracking!"
    
    print("\n--- 6 equipos ---")
    tournament6 = generate_circle_method_tournament(6, shuffle=False, seed=42)
    print(f"Número de partidos: {len(get_games_list(tournament6))}")
    
    best_assign6, min_breaks6, max_non_breaks6 = brute_force_optimal(tournament6)
    print(f"Breaks óptimos (fuerza bruta): {min_breaks6}")
    print(f"Máx non-breaks: {max_non_breaks6}")
    print(f"Cota inferior: {6 - 2} = 4")
    print(f"Solución válida: {verify_solution(tournament6, best_assign6)}")
    
    print("\n--- 8 equipos ---")
    tournament8 = generate_circle_method_tournament(8, shuffle=False, seed=42)
    print(f"Número de partidos: {len(get_games_list(tournament8))}")
    print("Corriendo backtracking (puede tardar un poco)...")
    
    best_assign8, min_breaks8, max_non_breaks8 = backtracking_solver(tournament8)
    print(f"Breaks óptimos (backtracking): {min_breaks8}")
    print(f"Máx non-breaks: {max_non_breaks8}")
    print(f"Cota inferior: {8 - 2} = 6")
    print(f"Solución válida: {verify_solution(tournament8, best_assign8)}")

