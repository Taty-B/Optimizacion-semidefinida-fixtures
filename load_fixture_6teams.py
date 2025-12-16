#!/usr/bin/env python
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


def build_fixture():
    teams = [1, 2, 3, 4, 5, 6]
    rounds = [1, 2, 3, 4, 5]
    matches = {
        1: [(6, 1), (2, 5), (3, 4)],
        2: [(6, 2), (3, 1), (4, 5)],
        3: [(6, 3), (4, 2), (5, 1)],
        4: [(6, 4), (5, 3), (1, 2)],
        5: [(6, 5), (1, 4), (2, 3)],
    }
    return teams, rounds, matches


def load_graph(session, teams, rounds, matches):
    session.run("MATCH (n:Vertex) DETACH DELETE n")

    vertices = [{"key": f"{t}-{r}", "team": t, "round": r} for t in teams for r in rounds]
    session.run(
        """
        UNWIND $verts AS v
        MERGE (:Vertex {key: v.key, team: v.team, round: v.round})
        """,
        {"verts": vertices},
    )

    match_edges = []
    for rnd, games in matches.items():
        for a, b in games:
            match_edges.append(
                {"from": f"{a}-{rnd}", "to": f"{b}-{rnd}", "round": rnd}
            )
    session.run(
        """
        UNWIND $edges AS e
        MATCH (u:Vertex {key: e.from})
        MATCH (v:Vertex {key: e.to})
        MERGE (u)-[:MATCH {round: e.round}]->(v)
        """,
        {"edges": match_edges},
    )

    consec_edges = []
    for t in teams:
        for r in rounds[:-1]:
            consec_edges.append(
                {
                    "from": f"{t}-{r}",
                    "to": f"{t}-{r+1}",
                    "round_from": r,
                    "round_to": r + 1,
                    "team": t,
                }
            )
    session.run(
        """
        UNWIND $edges AS e
        MATCH (u:Vertex {key: e.from})
        MATCH (v:Vertex {key: e.to})
        MERGE (u)-[r:CONSEC]->(v)
        SET r.round_from = e.round_from,
            r.round_to   = e.round_to,
            r.team       = e.team
        """,
        {"edges": consec_edges},
    )


def main():
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USERNAME") or os.environ["NEO4J_USER"]
    pwd = os.environ["NEO4J_PASSWORD"]
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    teams, rounds, matches = build_fixture()

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        load_graph(session, teams, rounds, matches)
    driver.close()
    print("Grafo del fixture de 6 equipos cargado en Neo4j.")


if __name__ == "__main__":
    main()
