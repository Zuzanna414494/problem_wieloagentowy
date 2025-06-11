"""
Projekt 6: Problem wieloagentowy - Texas Hold'em
Główny plik do uruchomienia eksperymentów z rozszerzonym trainerem
"""

import os
import sys
import time
from pathlib import Path

# Tworzenie struktury folderów
def setup_directories():
    """Tworzy niezbędne foldery"""
    directories = ['models', 'logs', 'results', 'plots']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Struktura folderów utworzona")

def check_dependencies():
    """Sprawdza czy wszystkie biblioteki są zainstalowane"""
    required_packages = [
        'pettingzoo', 'stable_baselines3', 'matplotlib',
        'numpy', 'torch', 'gymnasium'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Brakujące pakiety:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Zainstaluj brakujące pakiety:")
        print("pip install stable-baselines3[extra] pettingzoo[classic] matplotlib torch")
        return False

    print("✅ Wszystkie zależności są zainstalowane")
    return True

def main():
    """Główna funkcja projektu z rozszerzonym trainerem"""
    print("=" * 80)
    print("🎯 PROJEKT WIELOAGENTOWY - TEXAS HOLD'EM (ROZSZERZONY)")
    print("=" * 80)

    # Sprawdzenie zależności
    if not check_dependencies():
        return

    # Tworzenie struktury folderów
    setup_directories()

    # Import modułów projektu
    try:
        from environment_wrapper import TexasHoldemWrapper
        from algorithms_config import AlgorithmsConfig
        from trainer import EnhancedMultiAgentTrainer
        from visualizer import ResultsVisualizer

        print("✅ Wszystkie moduły załadowane pomyślnie")
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print("💡 Upewnij się, że plik paste.py zawiera klasę EnhancedMultiAgentTrainer")
        return

    # Konfiguracja eksperymentu
    print("\n🔧 Konfiguracja eksperymentu:")
    config = {
        'n_players': 4,              # 4 graczy w Texas Hold'em
        'total_timesteps': 20000,    # Zwiększona liczba kroków dla lepszej jakości
        'eval_freq': 5000,           # Częstotliwość ewaluacji
        'n_eval_episodes': 10,       # Więcej epizodów dla stabilniejszych wyników
        'max_combinations': 8        # Maksymalna liczba kombinacji heterogenicznych
    }

    for key, value in config.items():
        print(f"   {key}: {value}")

    # Inicjalizacja rozszerzonego trenera
    print("\n🤖 Inicjalizacja rozszerzonego trenera...")
    trainer = EnhancedMultiAgentTrainer(
        n_players=config['n_players'],
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        verbose=1
    )

    # Pobieranie konfiguracji algorytmów
    algorithms_config = AlgorithmsConfig.get_algorithms()
    print(f"📋 Skonfigurowane algorytmy: {list(algorithms_config.keys())}")

    # FAZA 1: Trening scenariuszy homogenicznych
    print("\n" + "="*60)
    print("🔄 FAZA 1: SCENARIUSZE HOMOGENICZNE")
    print("="*60)

    start_time = time.time()

    trainer.train_homogeneous_scenario(
        algorithm_configs=algorithms_config,
        total_timesteps=config['total_timesteps']
    )

    homo_time = time.time() - start_time
    print(f"⏱️  Czas treningu homogenicznych: {homo_time/60:.1f} minut")

    # FAZA 2: Trening scenariuszy heterogenicznych
    print("\n" + "="*60)
    print("🎯 FAZA 2: SCENARIUSZE HETEROGENICZNE")
    print("="*60)

    hetero_start_time = time.time()

    trainer.train_heterogeneous_scenarios(
        algorithm_configs=algorithms_config,
        total_timesteps=config['total_timesteps'] // 2,  # Mniej kroków dla każdego agenta
        max_combinations=config['max_combinations']
    )

    hetero_time = time.time() - hetero_start_time
    print(f"⏱️  Czas treningu heterogenicznych: {hetero_time/60:.1f} minut")

    total_time = time.time() - start_time
    print(f"⏱️  Całkowity czas treningu: {total_time/60:.1f} minut")

    # FAZA 3: Porównanie wszystkich scenariuszy
    print("\n" + "="*60)
    print("📊 FAZA 3: ANALIZA WSZYSTKICH WYNIKÓW")
    print("="*60)

    trainer.compare_all_scenarios()

    # FAZA 4: Tworzenie wizualizacji (jeśli visualizer obsługuje nowy format)
    print("\n📈 Tworzenie wykresów...")
    try:
        # Przygotowanie danych dla wizualizatora
        combined_results = {}
        combined_results.update(trainer.homogeneous_results)
        combined_results.update(trainer.heterogeneous_results)

        if combined_results:
            visualizer = ResultsVisualizer(combined_results)
            visualizer.create_all_plots()
            print("✅ Wykresy zostały wygenerowane")
        else:
            print("⚠️  Brak wyników do wizualizacji")

    except Exception as e:
        print(f"⚠️  Błąd podczas tworzenia wykresów: {e}")
        print("💡 Visualizer może wymagać aktualizacji dla nowego formatu danych")

    # FAZA 5: Zapisanie wyników
    print("\n💾 Zapisywanie wyników...")
    trainer.save_all_results('results/enhanced_experiment_results.pkl')

    # FAZA 6: Podsumowanie końcowe
    print("\n" + "="*80)
    print("🎉 ROZSZERZONY EKSPERYMENT ZAKOŃCZONY POMYŚLNIE!")
    print("="*80)

    # Wyświetlenie podsumowania
    trainer.print_experiment_summary()

    print("\n📁 Wygenerowane pliki:")
    print("   📊 plots/ - Wykresy i wizualizacje")
    print("   🤖 models/ - Wytrenowane modele")
    print("   📝 logs/ - Logi treningu")
    print("   💾 results/enhanced_experiment_results.pkl - Rozszerzone dane eksperymentu")

    # Najlepsze scenariusze z każdej kategorii
    best_homo = trainer.get_best_homogeneous_scenario()
    best_hetero = trainer.get_best_heterogeneous_scenario()

    print("\n🏆 NAJLEPSZE SCENARIUSZE:")
    print("-" * 50)

    if best_homo:
        homo_reward = trainer.homogeneous_results[best_homo]['mean_reward']
        print(f"🔄 Homogeniczny: {best_homo}")
        print(f"   Nagroda: {homo_reward:.3f}")
        print(f"   Agenci: {trainer.homogeneous_results[best_homo]['agents_config']}")

    if best_hetero:
        hetero_reward = trainer.heterogeneous_results[best_hetero]['mean_reward']
        print(f"🎯 Heterogeniczny: {best_hetero}")
        print(f"   Nagroda: {hetero_reward:.3f}")
        print(f"   Agenci: {trainer.heterogeneous_results[best_hetero]['agents_config']}")

    # Ogólne wnioski
    print("\n📊 WNIOSKI:")
    print("-" * 30)

    total_scenarios = len(trainer.homogeneous_results) + len(trainer.heterogeneous_results)
    print(f"🔢 Przetestowano {total_scenarios} różnych scenariuszy")

    if best_homo and best_hetero:
        homo_best_reward = trainer.homogeneous_results[best_homo]['mean_reward']
        hetero_best_reward = trainer.heterogeneous_results[best_hetero]['mean_reward']

        if hetero_best_reward > homo_best_reward:
            diff = hetero_best_reward - homo_best_reward
            print(f"🎯 Scenariusze heterogeniczne okazały się lepsze o {diff:.3f}")
            print("💡 Różnorodność algorytmów może przynosić korzyści!")
        elif homo_best_reward > hetero_best_reward:
            diff = homo_best_reward - hetero_best_reward
            print(f"🔄 Scenariusze homogeniczne okazały się lepsze o {diff:.3f}")
            print("💡 Spójność strategii może być kluczowa!")
        else:
            print("⚖️  Oba typy scenariuszy dały podobne wyniki")

    print(f"\n⏱️  Całkowity czas eksperymentu: {total_time/60:.1f} minut")

if __name__ == "__main__":
    main()