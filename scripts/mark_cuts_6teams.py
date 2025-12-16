#!/usr/bin/env python
import os
import sys

import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def compute_edge_cuts(tournament: Tournament):
    sdp_result, best_result, _ = solve_break_minimization(
        tournament, n_rounds=300, solver="SCS", seed=123
    )
    ha = best_result.assignment.home_away

    match_edges = []
    n_slots = tournament.n_slots
    for s in range(n_slots):
        round_num = s + 1
        seen = set()
        for i in range(tournament.n_teams):
            j = tournament.chart[i, s]
            if i < j and (i, j) not in seen:
                match_edges.append(
                    {
                        "from": f"{i+1}-{round_num}",
                        "to": f"{j+1}-{round_num}",
                        "round": round_num,
                        "cut": True,
                    }
                )
                seen.add((i, j))
                seen.add((j, i))

    consec_edges = []
    for i in range(tournament.n_teams):
        for s in range(1, n_slots):
            round_from = s
            round_to = s + 1
            cut = ha[i, s] != ha[i, s - 1]
            consec_edges.append(
                {
                    "from": f"{i+1}-{round_from}",
                    "to": f"{i+1}-{round_to}",
                    "round_from": round_from,
                    "round_to": round_to,
                    "team": i + 1,
                    "cut": bool(cut),
                }
            )

    node_roles = []
    for i in range(tournament.n_teams):
        for s in range(n_slots):
            round_num = s + 1
            is_home = bool(ha[i, s])
            node_roles.append(
                {
                    "key": f"{i+1}-{round_num}",
                    "home": is_home,
                    "role": "H" if is_home else "A",
                    "ha": int(is_home),
                }
            )

    breaks_total, hh, aa = count_breaks(tournament, best_result.assignment)
    return (
        match_edges,
        consec_edges,
        node_roles,
        best_result.assignment,
        {"total": breaks_total, "hh": hh, "aa": aa},
        sdp_result.sdp_value,
    )


def write_cuts_to_neo4j(match_edges, consec_edges, node_roles):
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USERNAME") or os.environ["NEO4J_USER"]
    pwd = os.environ["NEO4J_PASSWORD"]
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        session.run("MATCH ()-[r:UNCUT_EDGE]->() DELETE r")

        session.run(
            """
            UNWIND $edges AS e
            MATCH (u:Vertex {key: e.from})-[r:MATCH]-(v:Vertex {key: e.to})
            SET r.cut = e.cut
            """,
            {"edges": match_edges},
        )
        session.run(
            """
            UNWIND $edges AS e
            MATCH (u:Vertex {key: e.from})-[r:CONSEC]-(v:Vertex {key: e.to})
            SET r.cut       = e.cut,
                r.team      = e.team,
                r.round_from = e.round_from,
                r.round_to   = e.round_to
            """,
            {"edges": consec_edges},
        )

        uncut = [e for e in consec_edges if not e["cut"]]
        session.run(
            """
            UNWIND $edges AS e
            MATCH (u:Vertex {key: e.from})
            MATCH (v:Vertex {key: e.to})
            MERGE (u)-[:UNCUT_EDGE {
                team: e.team,
                round_from: e.round_from,
                round_to: e.round_to
            }]->(v)
            """,
            {"edges": uncut},
        )

        session.run(
            """
            UNWIND $nodes AS n
            MATCH (v:Vertex {key: n.key})
            SET v.home = n.home,
                v.role = n.role,
                v.ha   = n.ha
            """,
            {"nodes": node_roles},
        )
    driver.close()


def main():
    tournament = build_fixture_tournament()
    print("Tabla del torneo:")
    print(display_tournament_chart(tournament))

    (
        match_edges,
        consec_edges,
        node_roles,
        assignment,
        breaks_info,
        sdp_value,
    ) = compute_edge_cuts(tournament)

    print("\nMejor asignación (H=1, A=0):")
    print(display_assignment(tournament, assignment))
    print(
        f"\nBreaks totales: {breaks_info['total']} (HH: {breaks_info['hh']}, AA: {breaks_info['aa']})"
    )
    print(f"Valor SDP (cota superior de non-breaks): {sdp_value:.4f}")

    print("\nEscribiendo flags de corte en Neo4j...")
    write_cuts_to_neo4j(match_edges, consec_edges, node_roles)
    print("Listo.")


if __name__ == "__main__":
    main()
