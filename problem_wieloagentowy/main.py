"""
Projekt 6: Problem wieloagentowy - Texas Hold'em
Główny plik do uruchomienia eksperymentów
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
    """Główna funkcja projektu"""
    print("=" * 60)
    print("🎯 PROJEKT WIELOAGENTOWY - TEXAS HOLD'EM")
    print("=" * 60)
    
    # Sprawdzenie zależności
    if not check_dependencies():
        return
    
    # Tworzenie struktury folderów
    setup_directories()
    
    # Import modułów projektu
    try:
        from environment_wrapper import TexasHoldemWrapper
        from algorithms_config import AlgorithmsConfig
        from trainer import MultiAgentTrainer
        from visualizer import ResultsVisualizer
        
        print("✅ Wszystkie moduły załadowane pomyślnie")
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        return
    
    # Konfiguracja eksperymentu
    print("\n🔧 Konfiguracja eksperymentu:")
    config = {
        'n_players': 4,           # 4 graczy w Texas Hold'em
        'total_timesteps': 5000, # Liczba kroków treningu
        'eval_freq': 1000,        # Częstotliwość ewaluacji
        'n_eval_episodes': 5     # Liczba epizodów do ewaluacji
    }
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Inicjalizacja trenera
    print("\n🤖 Inicjalizacja trenera...")
    trainer = MultiAgentTrainer(
        n_players=config['n_players'],
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes']
    )
    
    # Pobieranie konfiguracji algorytmów
    algorithms_config = AlgorithmsConfig.get_algorithms()
    print(f"📋 Skonfigurowane algorytmy: {list(algorithms_config.keys())}")
    
    # Trening wszystkich algorytmów
    print("\n🚀 Rozpoczęcie treningu algorytmów...")
    start_time = time.time()
    
    for i, (alg_name, config_data) in enumerate(algorithms_config.items(), 1):
        print(f"\n[{i}/{len(algorithms_config)}] Trenowanie: {alg_name}")
        print("-" * 40)
        
        try:
            trainer.train_algorithm(
                algorithm_name=alg_name,
                algorithm_class=config_data["class"],
                hyperparams=config_data["params"],
                total_timesteps=config['total_timesteps']
            )
            print(f"✅ {alg_name} wytrenowany pomyślnie")
            
        except Exception as e:
            print(f"❌ Błąd podczas treningu {alg_name}: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Całkowity czas treningu: {total_time/60:.1f} minut")
    
    # Analiza wyników
    print("\n📊 Analiza wyników...")
    trainer.compare_algorithms()
    
    # Tworzenie wizualizacji
    print("\n📈 Tworzenie wykresów...")
    visualizer = ResultsVisualizer(trainer.results)
    visualizer.create_all_plots()
    
    # Zapisanie wyników
    print("\n💾 Zapisywanie wyników...")
    trainer.save_results('results/experiment_results.pkl')
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("🎉 EKSPERYMENT ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 60)
    print("📁 Wygenerowane pliki:")
    print("   📊 plots/ - Wykresy i wizualizacje")
    print("   🤖 models/ - Wytrenowane modele")
    print("   📝 logs/ - Logi treningu")
    print("   💾 results/ - Dane eksperymentu")
    
    # Wyświetlenie najlepszego algorytmu
    if trainer.results:
        best_alg = max(trainer.results.items(), 
                      key=lambda x: x[1]['mean_reward'])
        print(f"\n🏆 Najlepszy algorytm: {best_alg[0]}")
        print(f"   Średnia nagroda: {best_alg[1]['mean_reward']:.2f}")

if __name__ == "__main__":
    main()