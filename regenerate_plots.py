#!/usr/bin/env python
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from tournament import (
    Tournament, Assignment, generate_circle_method_tournament,
    count_breaks, count_non_breaks, display_tournament_chart,
    display_assignment, get_games_list, get_consecutive_pairs
)
from sdp_solver import solve_break_minimization
from exact_solver import brute_force_optimal, backtracking_solver, verify_solution

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_experiments():
    print("Corriendo experimentos...")
    team_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    n_rounds_per_instance = 500
    n_instances = 3

    results = []
    for n_teams in team_sizes:
        print(f"  {n_teams} equipos...", end=" ", flush=True)
        instance_results = []
        for instance in range(n_instances):
            tournament = generate_circle_method_tournament(
                n_teams, shuffle=True, seed=42 + instance * 100
            )
            sdp_result, best_result, all_round_results = solve_break_minimization(
                tournament, n_rounds=n_rounds_per_instance, solver='SCS', seed=42 + instance
            )
            all_non_breaks = [r.non_breaks for r in all_round_results]
            all_ratios = [r.objective_ratio for r in all_round_results]
            instance_results.append({
                'sdp_value': sdp_result.sdp_value,
                'sdp_time': sdp_result.solve_time,
                'best_breaks': best_result.breaks,
                'best_non_breaks': best_result.non_breaks,
                'best_ratio': best_result.objective_ratio,
                'avg_non_breaks': np.mean(all_non_breaks),
                'avg_ratio': np.mean(all_ratios)
            })

        lower_bound = n_teams - 2

        results.append({
            'n_teams': n_teams,
            'n_games': len(get_games_list(tournament)),
            'n_pairs': len(get_consecutive_pairs(tournament)),
            'lower_bound': lower_bound,
            'sdp_value': np.mean([r['sdp_value'] for r in instance_results]),
            'sdp_time': np.mean([r['sdp_time'] for r in instance_results]),
            'best_breaks': np.mean([r['best_breaks'] for r in instance_results]),
            'best_non_breaks': np.mean([r['best_non_breaks'] for r in instance_results]),
            'best_ratio': np.mean([r['best_ratio'] for r in instance_results]),
            'avg_ratio': np.mean([r['avg_ratio'] for r in instance_results])
        })
        print(f"breaks={results[-1]['best_breaks']:.1f}, ratio={results[-1]['avg_ratio']:.4f}")

    return pd.DataFrame(results)


def plot_ratio_time(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(df['n_teams'], df['avg_ratio'], 'bo-', linewidth=2, markersize=8, label='Ratio promedio')
    ax1.plot(df['n_teams'], df['best_ratio'], 'g^--', linewidth=2, markersize=8, label='Mejor ratio')
    ax1.axhline(y=0.87856, color='r', linestyle=':', linewidth=2, label='Garantía teórica (0.878)')
    ax1.set_xlabel('Número de Equipos')
    ax1.set_ylabel('Ratio de Aproximación')
    ax1.set_title('Ratio de Aproximación vs Tamaño del Torneo')
    ax1.legend()
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df['n_teams'], df['sdp_time'], 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Equipos')
    ax2.set_ylabel('Tiempo (segundos)')
    ax2.set_title('Tiempo de Resolución SDP')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'results_ratio_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardado: results_ratio_time.png")


def plot_breaks_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df['n_teams']
    width = 0.35

    bars1 = ax.bar(x - width/2, df['best_breaks'], width, label='Breaks (SDP+GW)', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x + width/2, df['lower_bound'], width, label='Cota Inferior (n-2)', color='lightcoral', edgecolor='darkred')

    ax.set_xlabel('Número de Equipos')
    ax.set_ylabel('Número de Breaks')
    ax.set_title('Comparación: Breaks Obtenidos vs Cota Inferior Teórica (n-2)')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'results_breaks_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardado: results_breaks_comparison.png")


def plot_distribution_16teams():
    print("  Generando distribución 16 equipos (1000 rondas)...")
    tournament_16 = generate_circle_method_tournament(16, shuffle=True, seed=42)
    sdp_result_16, best_16, all_results_16 = solve_break_minimization(
        tournament_16, n_rounds=1000, solver='SCS', seed=42
    )

    all_breaks_16 = [r.breaks for r in all_results_16]
    all_ratios_16 = [r.objective_ratio for r in all_results_16]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(all_breaks_16, bins=range(min(all_breaks_16), max(all_breaks_16)+2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=best_16.breaks, color='red', linestyle='--', linewidth=2,
                label=f'Mejor: {best_16.breaks}')
    ax1.axvline(x=np.mean(all_breaks_16), color='green', linestyle=':', linewidth=2,
                label=f'Promedio: {np.mean(all_breaks_16):.1f}')
    ax1.set_xlabel('Número de Breaks')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Breaks (16 equipos, 1000 rondas)')
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(all_ratios_16, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(x=0.87856, color='red', linestyle='--', linewidth=2, label='Garantía GW: 0.878')
    ax2.axvline(x=np.mean(all_ratios_16), color='green', linestyle=':', linewidth=2,
                label=f'Promedio: {np.mean(all_ratios_16):.4f}')
    ax2.set_xlabel('Ratio de Aproximación')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Ratios de Aproximación')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'results_distribution_16teams.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardado: results_distribution_16teams.png")


def plot_scalability(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['n_teams'], df['sdp_time'], 'bo-', linewidth=2, markersize=8, label='Tiempo real')

    coeffs = np.polyfit(df['n_teams'], df['sdp_time'], 3)
    x_fit = np.linspace(4, 20, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label='Ajuste cúbico')

    ax.set_xlabel('Número de Equipos')
    ax.set_ylabel('Tiempo (segundos)')
    ax.set_title('Escalabilidad del Algoritmo SDP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'results_scalability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardado: results_scalability.png")


def plot_gap_analysis(df):
    df['gap_to_lower'] = df['best_breaks'] - df['lower_bound']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['n_teams'], df['gap_to_lower'], color='orange', edgecolor='darkorange')
    ax.set_xlabel('Número de Equipos')
    ax.set_ylabel('Gap (Breaks - Cota Inferior)')
    ax.set_title('Gap entre Solución SDP y Cota Inferior Teórica (n-2)')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (x, y) in enumerate(zip(df['n_teams'], df['gap_to_lower'])):
        ax.annotate(f'{int(y)}', xy=(x, y), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'results_gap_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardado: results_gap_analysis.png")


def plot_assignment_8teams():
    print("  Generando gráfico de asignación 8 equipos...")
    tournament_8 = generate_circle_method_tournament(8, shuffle=False, seed=42)
    sdp_result, best_result, _ = solve_break_minimization(
        tournament_8, n_rounds=500, solver='SCS', seed=42
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    n = tournament_8.n_teams
    n_slots = tournament_8.n_slots
    ha = best_result.assignment.home_away
    colors = np.where(ha == 1, 1, 0)
    im = ax.imshow(colors, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.grid(False)

    for i in range(n):
        for s in range(1, n_slots):
            if ha[i, s-1] == ha[i, s]:
                rect = plt.Rectangle((s-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(rect)

    for i in range(n):
        for s in range(n_slots):
            label = 'H' if ha[i, s] == 1 else 'A'
            color = 'white' if ha[i, s] == 1 else 'black'
            ax.text(s, i, label, ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    ax.set_xticks(range(n_slots))
    ax.set_xticklabels([f'F{s+1}' for s in range(n_slots)])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'Eq {i+1}' for i in range(n)])
    ax.set_title(f"Mejor Asignación Encontrada ({best_result.breaks} breaks)\n(Recuadros rojos = breaks)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Equipo')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Local (H)'),
        Patch(facecolor='red', label='Visitante (A)'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='Break')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    filename = f"asignacion_optima_{n}equipos_{best_result.breaks}breaks.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {filename}")


def main():
    print("=" * 60)
    print("Regenerando gráficos con cota inferior corregida (n-2)")
    print("=" * 60)

    df = run_experiments()

    print("\nGenerando gráficos...")
    plot_ratio_time(df)
    plot_breaks_comparison(df)
    plot_scalability(df)
    plot_gap_analysis(df)
    plot_distribution_16teams()
    plot_assignment_8teams()

    print("\n" + "=" * 60)
    print("¡Todos los gráficos fueron regenerados correctamente!")
    print("=" * 60)

    print("\nResumen de resultados (cota inferior corregida n-2):")
    summary = df[['n_teams', 'best_breaks', 'lower_bound', 'avg_ratio', 'sdp_time']].copy()
    summary.columns = ['Teams', 'Breaks', 'Lower Bound', 'Avg Ratio', 'Time (s)']
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
