"""
Rozszerzony moduł do treningu i ewaluacji algorytmów uczenia ze wzmocnieniem
w środowisku Texas Hold'em z obsługą różnych scenariuszy wieloagentowych
"""

import os
import time
import pickle
import numpy as np
from typing import Dict, Any, Type, Optional, List, Union
from pathlib import Path
import itertools

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch

from environment_wrapper import make_env

class EnhancedMultiAgentTrainer:
    """Klasa do treningu i porównywania algorytmów wieloagentowych z różnymi scenariuszami"""

    def __init__(self, n_players: int = 4, eval_freq: int = 5000,
                 n_eval_episodes: int = 5, verbose: int = 1):
        """
        Inicjalizuje trainer

        Args:
            n_players: Liczba graczy w Texas Hold'em
            eval_freq: Częstotliwość ewaluacji podczas treningu
            n_eval_episodes: Liczba epizodów do ewaluacji
            verbose: Poziom szczegółowości logów
        """
        self.n_players = n_players
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.results = {}
        self.homogeneous_results = {}  # Wszyscy agenci ten sam algorytm
        self.heterogeneous_results = {}  # Różni agenci różne algorytmy

        # Informacje o systemie
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose > 0:
            print(f"🖥️  Używane urządzenie: {self.device}")

    def train_homogeneous_scenario(self, algorithm_configs: Dict[str, Dict[str, Any]],
                                 total_timesteps: int = 100000):
        """
        Trenuje scenariusze gdzie wszyscy agenci używają tego samego algorytmu

        Args:
            algorithm_configs: Słownik z konfiguracjami algorytmów
            total_timesteps: Liczba kroków treningu dla każdego algorytmu
        """
        print("\n🔄 SCENARIUSZ HOMOGENICZNY - Wszyscy agenci ten sam algorytm")
        print("=" * 60)

        for alg_name, config in algorithm_configs.items():
            print(f"\n🤖 Trenowanie scenariusza: Wszyscy agenci = {alg_name}")

            try:
                # Tworzymy jeden model który będzie reprezentować wszystkich agentów
                train_env = DummyVecEnv([make_env(self.n_players, seed=42)])
                eval_env = DummyVecEnv([make_env(self.n_players, seed=123)])

                start_time = time.time()

                # Inicjalizacja modelu
                model = config["class"](
                    "MlpPolicy",
                    train_env,
                    verbose=self.verbose,
                    device=self.device,
                    **config["params"]
                )

                # Trening
                model.learn(total_timesteps=total_timesteps)
                training_time = time.time() - start_time

                # Ewaluacja
                mean_reward, std_reward = evaluate_policy(
                    model, eval_env, n_eval_episodes=self.n_eval_episodes
                )

                # Zapisanie wyników
                scenario_name = f"HOMO_{alg_name}"
                self.homogeneous_results[scenario_name] = {
                    'algorithm': alg_name,
                    'scenario_type': 'homogeneous',
                    'agents_config': [alg_name] * self.n_players,
                    'model': model,
                    'mean_reward': float(mean_reward),
                    'std_reward': float(std_reward),
                    'training_time': training_time,
                    'hyperparams': config["params"].copy(),
                    'total_timesteps': total_timesteps,
                    'n_players': self.n_players
                }

                train_env.close()
                eval_env.close()

                print(f"✅ {scenario_name}: {mean_reward:.3f}±{std_reward:.3f} "
                      f"(czas: {training_time/60:.1f}min)")

            except Exception as e:
                print(f"❌ Błąd w scenariuszu {alg_name}: {e}")

    def train_heterogeneous_scenarios(self, algorithm_configs: Dict[str, Dict[str, Any]],
                                    total_timesteps: int = 100000,
                                    max_combinations: int = 10):
        """
        Trenuje scenariusze gdzie różni agenci używają różnych algorytmów

        Args:
            algorithm_configs: Słownik z konfiguracjami algorytmów
            total_timesteps: Liczba kroków treningu dla każdego agenta
            max_combinations: Maksymalna liczba kombinacji do przetestowania
        """
        print("\n🎯 SCENARIUSZ HETEROGENICZNY - Różni agenci różne algorytmy")
        print("=" * 60)

        # Generowanie kombinacji algorytmów
        algorithm_names = list(algorithm_configs.keys())

        # Różne strategie kombinacji
        combinations = []

        # 1. Wszystkie pary algorytmów (dla 2 graczy każdy)
        if self.n_players == 4:
            for alg1, alg2 in itertools.combinations(algorithm_names, 2):
                combinations.append([alg1, alg1, alg2, alg2])

        # 2. Jeden algorytm vs reszta innych
        for main_alg in algorithm_names:
            other_algs = [alg for alg in algorithm_names if alg != main_alg]
            if len(other_algs) >= self.n_players - 1:
                config = [main_alg] + other_algs[:self.n_players-1]
                combinations.append(config)

        # 3. Równomierne mieszanie (jeśli mamy wystarczająco algorytmów)
        if len(algorithm_names) >= self.n_players:
            combinations.append(algorithm_names[:self.n_players])

        # Ograniczenie liczby kombinacji
        combinations = combinations[:max_combinations]

        print(f"🔄 Testowanie {len(combinations)} kombinacji heterogenicznych...")

        for i, agent_algorithms in enumerate(combinations, 1):
            scenario_name = f"HETERO_{i:02d}_{'_'.join(agent_algorithms)}"
            print(f"\n[{i}/{len(combinations)}] {scenario_name}")
            print(f"   Agenci: {agent_algorithms}")

            try:
                result = self._train_heterogeneous_combination(
                    agent_algorithms, algorithm_configs, total_timesteps, scenario_name
                )

                self.heterogeneous_results[scenario_name] = result

                print(f"✅ {scenario_name}: {result['mean_reward']:.3f}±{result['std_reward']:.3f}")

            except Exception as e:
                print(f"❌ Błąd w kombinacji {scenario_name}: {e}")

    def _train_heterogeneous_combination(self, agent_algorithms: List[str],
                                       algorithm_configs: Dict[str, Dict[str, Any]],
                                       total_timesteps: int, scenario_name: str):
        """
        Trenuje jedną konkretną kombinację heterogeniczną
        """
        models = []
        training_times = []

        # Trenowanie każdego agenta osobno
        for agent_id, alg_name in enumerate(agent_algorithms):
            config = algorithm_configs[alg_name]

            # Środowisko dla pojedynczego agenta (uproszczone do 2 graczy dla szybszego treningu)
            train_env = DummyVecEnv([make_env(n_players=2, seed=42 + agent_id)])

            start_time = time.time()

            # Inicjalizacja i trening modelu
            model = config["class"](
                "MlpPolicy",
                train_env,
                verbose=0,  # Mniej verbose dla heterogenicznych
                device=self.device,
                **config["params"]
            )

            model.learn(total_timesteps=total_timesteps)
            training_time = time.time() - start_time

            models.append(model)
            training_times.append(training_time)
            train_env.close()

        # Ewaluacja całej kombinacji
        eval_rewards = self._evaluate_heterogeneous_combination(models, agent_algorithms)

        return {
            'algorithm': 'Mixed',
            'scenario_type': 'heterogeneous',
            'agents_config': agent_algorithms,
            'models': models,
            'mean_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'training_time': float(np.sum(training_times)),
            'individual_training_times': training_times,
            'total_timesteps': total_timesteps * len(agent_algorithms),
            'n_players': self.n_players,
            'individual_rewards': eval_rewards
        }

    def _evaluate_heterogeneous_combination(self, models: List[BaseAlgorithm],
                                          agent_algorithms: List[str]) -> List[float]:
        """
        Ewaluuje kombinację heterogeniczną poprzez symulację gier
        """
        rewards = []

        # Prosta ewaluacja - każdy model gra przeciwko prostemu środowisku
        for i, model in enumerate(models):
            eval_env = DummyVecEnv([make_env(n_players=2, seed=100 + i)])

            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = eval_env.reset()
                total_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = eval_env.step(action)
                    total_reward += reward[0]

                episode_rewards.append(total_reward)

            rewards.append(np.mean(episode_rewards))
            eval_env.close()

        return rewards

    def compare_all_scenarios(self):
        """Porównuje wszystkie scenariusze (homogeniczne i heterogeniczne)"""
        print("\n" + "="*80)
        print("📊 PORÓWNANIE WSZYSTKICH SCENARIUSZY")
        print("="*80)

        all_results = {}
        all_results.update(self.homogeneous_results)
        all_results.update(self.heterogeneous_results)

        if not all_results:
            print("❌ Brak wyników do porównania")
            return

        # Sortowanie według średniej nagrody
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True
        )

        print("🏆 RANKING SCENARIUSZY:")
        print("-" * 80)
        print(f"{'Ranga':<5} {'Scenariusz':<25} {'Typ':<12} {'Nagroda':<15} {'Czas [min]':<10}")
        print("-" * 80)

        for i, (name, results) in enumerate(sorted_results, 1):
            mean_reward = results['mean_reward']
            std_reward = results['std_reward']
            training_time = results['training_time'] / 60
            scenario_type = results['scenario_type']

            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."

            print(f"{medal:<5} {name:<25} {scenario_type:<12} "
                  f"{mean_reward:>6.3f}±{std_reward:<5.3f} {training_time:>8.1f}")

        print("-" * 80)

        # Analiza typów scenariuszy
        self._analyze_scenario_types()

    def _analyze_scenario_types(self):
        """Analizuje różnice między scenariuszami homogenicznymi a heterogenicznymi"""
        print("\n🔍 ANALIZA TYPÓW SCENARIUSZY:")
        print("-" * 50)

        homo_rewards = [r['mean_reward'] for r in self.homogeneous_results.values()]
        hetero_rewards = [r['mean_reward'] for r in self.heterogeneous_results.values()]

        if homo_rewards and hetero_rewards:
            homo_mean = np.mean(homo_rewards)
            homo_std = np.std(homo_rewards)
            hetero_mean = np.mean(hetero_rewards)
            hetero_std = np.std(hetero_rewards)

            print(f"📈 Scenariusze HOMOGENICZNE:")
            print(f"   Średnia nagroda: {homo_mean:.3f} ± {homo_std:.3f}")
            print(f"   Liczba scenariuszy: {len(homo_rewards)}")
            print(f"   Najlepszy: {max(homo_rewards):.3f}")

            print(f"\n🎯 Scenariusze HETEROGENICZNE:")
            print(f"   Średnia nagroda: {hetero_mean:.3f} ± {hetero_std:.3f}")
            print(f"   Liczba scenariuszy: {len(hetero_rewards)}")
            print(f"   Najlepszy: {max(hetero_rewards):.3f}")

            print(f"\n📊 PORÓWNANIE:")
            difference = hetero_mean - homo_mean
            if abs(difference) > 0.01:
                better = "HETEROGENICZNE" if difference > 0 else "HOMOGENICZNE"
                print(f"   🏆 Lepsze: {better} (różnica: {abs(difference):.3f})")
            else:
                print(f"   ⚖️  Podobne wyniki (różnica: {abs(difference):.3f})")

    def get_best_homogeneous_scenario(self) -> Optional[str]:
        """Zwraca najlepszy scenariusz homogeniczny"""
        if not self.homogeneous_results:
            return None
        return max(self.homogeneous_results.items(),
                  key=lambda x: x[1]['mean_reward'])[0]

    def get_best_heterogeneous_scenario(self) -> Optional[str]:
        """Zwraca najlepszy scenariusz heterogeniczny"""
        if not self.heterogeneous_results:
            return None
        return max(self.heterogeneous_results.items(),
                  key=lambda x: x[1]['mean_reward'])[0]

    def save_all_results(self, filepath: str = "results/enhanced_experiment_results.pkl"):
        """Zapisuje wszystkie wyniki eksperymentu"""
        try:
            save_data = {
                'homogeneous': {},
                'heterogeneous': {},
                'metadata': {
                    'experiment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device': self.device,
                    'n_players': self.n_players,
                    'eval_freq': self.eval_freq,
                    'n_eval_episodes': self.n_eval_episodes
                }
            }

            # Zapisz homogeniczne (bez modeli)
            for name, results in self.homogeneous_results.items():
                save_data['homogeneous'][name] = {
                    k: v for k, v in results.items() if k != 'model'
                }

            # Zapisz heterogeniczne (bez modeli)
            for name, results in self.heterogeneous_results.items():
                save_data['heterogeneous'][name] = {
                    k: v for k, v in results.items() if k != 'models'
                }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"💾 Wszystkie wyniki zapisane do: {filepath}")

        except Exception as e:
            print(f"❌ Błąd podczas zapisywania: {e}")

    def print_experiment_summary(self):
        """Wyświetla podsumowanie całego eksperymentu"""
        print("\n" + "="*60)
        print("📋 PODSUMOWANIE EKSPERYMENTU")
        print("="*60)

        total_homo = len(self.homogeneous_results)
        total_hetero = len(self.heterogeneous_results)

        print(f"🔄 Scenariusze homogeniczne: {total_homo}")
        print(f"🎯 Scenariusze heterogeniczne: {total_hetero}")
        print(f"📊 Łącznie scenariuszy: {total_homo + total_hetero}")

        if self.homogeneous_results:
            best_homo = self.get_best_homogeneous_scenario()
            best_homo_reward = self.homogeneous_results[best_homo]['mean_reward']
            print(f"\n🏆 Najlepszy homogeniczny: {best_homo}")
            print(f"   Nagroda: {best_homo_reward:.3f}")

        if self.heterogeneous_results:
            best_hetero = self.get_best_heterogeneous_scenario()
            best_hetero_reward = self.heterogeneous_results[best_hetero]['mean_reward']
            print(f"\n🎯 Najlepszy heterogeniczny: {best_hetero}")
            print(f"   Nagroda: {best_hetero_reward:.3f}")

        print("\n" + "="*60)

# Test modułu
if __name__ == "__main__":
    print("🧪 Test rozszerzonego modułu treningu...")

    # Przykładowa konfiguracja algorytmów do testów
    from stable_baselines3 import PPO, A2C

    test_algorithms = {
        "PPO_Test": {
            "class": PPO,
            "params": {
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 64
            }
        },
        "A2C_Test": {
            "class": A2C,
            "params": {
                "learning_rate": 7e-4,
                "n_steps": 5
            }
        }
    }

    # Test trenera
    trainer = EnhancedMultiAgentTrainer(n_players=4, verbose=1)

    print("✅ Rozszerzony trainer utworzony pomyślnie")
    print("🧪 Gotowy do testowania scenariuszy homogenicznych i heterogenicznych")