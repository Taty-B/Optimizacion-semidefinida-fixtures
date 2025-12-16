import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Tournament:
    n_teams: int
    chart: np.ndarray
    
    @property
    def n_slots(self) -> int:
        return self.n_teams - 1
    
    def validate(self) -> bool:
        n = self.n_teams
        T = self.chart
        
        if T.shape != (n, n - 1):
            return False
        
        for i in range(n):
            expected = set(range(n)) - {i}
            actual = set(T[i, :])
            if expected != actual:
                return False
        
        for i in range(n):
            for s in range(n - 1):
                j = T[i, s]
                if T[j, s] != i:
                    return False
        
        return True


@dataclass
class Assignment:
    home_away: np.ndarray
    
    @classmethod
    def from_game_directions(cls, tournament: Tournament, directions: np.ndarray) -> 'Assignment':
        n = tournament.n_teams
        n_slots = tournament.n_slots
        home_away = np.zeros((n, n_slots), dtype=int)
        
        game_idx = 0
        for s in range(n_slots):
            seen = set()
            for i in range(n):
                j = tournament.chart[i, s]
                if (i, j) not in seen and (j, i) not in seen:
                    if i < j:
                        if directions[game_idx] == 1:
                            home_away[i, s] = 1
                            home_away[j, s] = 0
                        else:
                            home_away[i, s] = 0
                            home_away[j, s] = 1
                    else:
                        if directions[game_idx] == 1:
                            home_away[j, s] = 1
                            home_away[i, s] = 0
                        else:
                            home_away[j, s] = 0
                            home_away[i, s] = 1
                    seen.add((i, j))
                    seen.add((j, i))
                    game_idx += 1
        
        return cls(home_away=home_away)


def generate_circle_method_tournament(n_teams: int, shuffle: bool = True, 
                                       seed: Optional[int] = None) -> Tournament:
    if n_teams % 2 != 0:
        raise ValueError("Number of teams must be even.")
    if n_teams < 2:
        raise ValueError("Need at least 2 teams.")
    
    rng = np.random.default_rng(seed)
    n_slots = n_teams - 1
    
    chart = np.zeros((n_teams, n_slots), dtype=int)
    
    circle = list(range(1, n_teams))
    
    for slot in range(n_slots):
        chart[0, slot] = circle[0]
        chart[circle[0], slot] = 0
        
        for k in range(1, n_teams // 2):
            i = circle[k]
            j = circle[n_teams - 1 - k]
            chart[i, slot] = j
            chart[j, slot] = i
        
        circle = circle[1:] + [circle[0]]
    
    if shuffle:
        perm = rng.permutation(n_teams)
        inv_perm = np.argsort(perm)
        
        new_chart = np.zeros_like(chart)
        for i in range(n_teams):
            for s in range(n_slots):
                new_chart[perm[i], s] = perm[chart[i, s]]
        chart = new_chart
    
    tournament = Tournament(n_teams=n_teams, chart=chart)
    assert tournament.validate(), "Generated tournament is invalid!"
    
    return tournament


def count_breaks(tournament: Tournament, assignment: Assignment) -> Tuple[int, int, int]:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    ha = assignment.home_away
    
    hh_breaks = 0
    aa_breaks = 0
    
    for i in range(n):
        for s in range(1, n_slots):
            if ha[i, s-1] == 1 and ha[i, s] == 1:
                hh_breaks += 1
            elif ha[i, s-1] == 0 and ha[i, s] == 0:
                aa_breaks += 1
    
    return hh_breaks + aa_breaks, hh_breaks, aa_breaks


def count_non_breaks(tournament: Tournament, assignment: Assignment) -> int:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    total_edges = n * (n_slots - 1)
    breaks, _, _ = count_breaks(tournament, assignment)
    return total_edges - breaks


def get_games_list(tournament: Tournament) -> List[Tuple[int, int, int]]:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    games = []
    
    for s in range(n_slots):
        seen = set()
        for i in range(n):
            j = tournament.chart[i, s]
            if i < j and (i, j) not in seen:
                games.append((i, j, s))
                seen.add((i, j))
    
    return games


def get_consecutive_pairs(tournament: Tournament) -> List[Tuple[int, int, int]]:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    pairs = []
    
    for i in range(n):
        for s in range(1, n_slots):
            pairs.append((i, s - 1, s))
    
    return pairs


def display_tournament_chart(tournament: Tournament) -> str:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    
    lines = ["Fechas: " + "  ".join(f"{s+1:2d}" for s in range(n_slots))]
    lines.append("-" * (8 + 4 * n_slots))
    
    for i in range(n):
        row = f"Eq {i+1:2d}: " + "  ".join(f"{tournament.chart[i, s]+1:2d}" for s in range(n_slots))
        lines.append(row)
    
    return "\n".join(lines)


def display_assignment(tournament: Tournament, assignment: Assignment) -> str:
    n = tournament.n_teams
    n_slots = tournament.n_slots
    ha = assignment.home_away
    
    lines = ["Fechas: " + " ".join(f"{s+1:2d}" for s in range(n_slots))]
    lines.append("-" * (8 + 3 * n_slots))
    
    for i in range(n):
        row = f"Eq {i+1:2d}: " + " ".join("H " if ha[i, s] == 1 else "A " for s in range(n_slots))
        lines.append(row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("Probando generación del torneo y conteo de breaks")
    print("=" * 50)
    
    tournament = generate_circle_method_tournament(8, shuffle=False, seed=42)
    print(f"\nTorneo con {tournament.n_teams} equipos, {tournament.n_slots} fechas:")
    print(display_tournament_chart(tournament))
    print(f"\nTorneo válido: {tournament.validate()}")
    
    n = tournament.n_teams
    n_slots = tournament.n_slots
    home_away = np.zeros((n, n_slots), dtype=int)
    
    for s in range(n_slots):
        for i in range(n):
            j = tournament.chart[i, s]
            if i < j:
                home_away[i, s] = 1
                home_away[j, s] = 0
    
    assignment = Assignment(home_away=home_away)
    print("\nAsignación simple (el equipo con índice menor juega de local):")
    print(display_assignment(tournament, assignment))
    
    breaks, hh, aa = count_breaks(tournament, assignment)
    non_breaks = count_non_breaks(tournament, assignment)
    print(f"\nBreaks: {breaks} (HH: {hh}, AA: {aa})")
    print(f"Non-breaks: {non_breaks}")
    print(f"Cota inferior de breaks: {n - 2} = {tournament.n_teams - 2}")

