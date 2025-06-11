"""
Moduł do treningu i ewaluacji algorytmów uczenia ze wzmocnieniem
w środowisku Texas Hold'em
"""

import os
import time
import pickle
import numpy as np
from typing import Dict, Any, Type, Optional
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch

from environment_wrapper import make_env

class MultiAgentTrainer:
    """Klasa do treningu i porównywania algorytmów wieloagentowych"""
    
    def __init__(self, n_players: int = 4, eval_freq: int = 5000, 
                 n_eval_episodes: int = 20, verbose: int = 1):
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
        
        # Informacje o systemie
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose > 0:
            print(f"🖥️  Używane urządzenie: {self.device}")
    
    def train_algorithm(self, algorithm_name: str, algorithm_class: Type[BaseAlgorithm],
                       hyperparams: Dict[str, Any], total_timesteps: int = 100000) -> BaseAlgorithm:
        """
        Trenuje pojedynczy algorytm
        
        Args:
            algorithm_name: Nazwa algorytmu
            algorithm_class: Klasa algorytmu (PPO, A2C, DQN)
            hyperparams: Hiperparametry algorytmu
            total_timesteps: Liczba kroków treningu
            
        Returns:
            Wytrenowany model
        """
        if self.verbose > 0:
            print(f"\n🚀 Rozpoczęcie treningu: {algorithm_name}")
            print(f"   Timesteps: {total_timesteps:,}")
            print(f"   Urządzenie: {self.device}")
        
        try:
            # Tworzenie środowisk
            train_env = DummyVecEnv([make_env(self.n_players, seed=42)])
            eval_env = DummyVecEnv([make_env(self.n_players, seed=123)])
            
            # Przygotowanie folderów
            model_path = Path(f"models/{algorithm_name}")
            log_path = Path(f"logs/{algorithm_name}")
            model_path.mkdir(parents=True, exist_ok=True)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Tworzenie modelu
            model = algorithm_class(
                "MlpPolicy",
                train_env,
                verbose=self.verbose,
                device=self.device,
                **hyperparams
            )
            
            if self.verbose > 0:
                print(f"✅ Model {algorithm_name} utworzony")
                total_params = sum(p.numel() for p in model.policy.parameters())
                print(f"   Parametry sieci: {total_params:,}")
            
            # Callback do ewaluacji
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(model_path / "best_model"),
                log_path=str(log_path),
                eval_freq=self.eval_freq,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose
            )
            
            # Callback do zapisywania checkpointów
            checkpoint_callback = CheckpointCallback(
                save_freq=self.eval_freq * 2,
                save_path=str(model_path / "checkpoints"),
                name_prefix=algorithm_name
            )
            
            # Trening
            start_time = time.time()
            
            if self.verbose > 0:
                print(f"⏱️  Rozpoczęcie treningu...")
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback],
                progress_bar=False,
                reset_num_timesteps=True
            )
            
            training_time = time.time() - start_time
            
            if self.verbose > 0:
                print(f"✅ Trening zakończony w {training_time/60:.1f} minut")
            
            # Ewaluacja końcowa
            if self.verbose > 0:
                print(f"📊 Ewaluacja końcowa...")
            
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes * 2,  # Więcej epizodów na końcu
                deterministic=True,
                render=False
            )
            
            # Zapisanie finalnego modelu
            final_model_path = model_path / "final_model"
            model.save(str(final_model_path))
            
            # Zapisanie wyników
            self.results[algorithm_name] = {
                'model': model,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'training_time': training_time,
                'hyperparams': hyperparams.copy(),
                'total_timesteps': total_timesteps,
                'n_players': self.n_players,
                'final_model_path': str(final_model_path)
            }
            
            if self.verbose > 0:
                print(f"🎯 Wyniki dla {algorithm_name}:")
                print(f"   Średnia nagroda: {mean_reward:.3f} ± {std_reward:.3f}")
                print(f"   Czas treningu: {training_time:.1f}s")
            
            # Zamknięcie środowisk
            train_env.close()
            eval_env.close()
            
            return model
            
        except Exception as e:
            print(f"❌ Błąd podczas treningu {algorithm_name}: {e}")
            if self.verbose > 1:
                import traceback
                traceback.print_exc()
            raise
    
    def compare_algorithms(self):
        """Porównuje wszystkie wytrenowane algorytmy"""
        if not self.results:
            print("❌ Brak wyników do porównania")
            return
        
        print("\n" + "="*60)
        print("📊 PORÓWNANIE ALGORYTMÓW")
        print("="*60)
        
        # Sortowanie według średniej nagrody
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True
        )
        
        print(f"{'Algorytm':<20} {'Nagroda':<15} {'Czas [min]':<12} {'Wydajność':<12}")
        print("-" * 60)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            mean_reward = results['mean_reward']
            std_reward = results['std_reward']
            training_time = results['training_time'] / 60
            efficiency = mean_reward / training_time if training_time > 0 else 0
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{medal} {name:<18} {mean_reward:>6.3f}±{std_reward:<5.3f} "
                  f"{training_time:>8.1f}    {efficiency:>8.4f}")
        
        print("-" * 60)
        
        # Szczegółowe porównanie hiperparametrów
        print("\n🔧 ANALIZA HIPERPARAMETRÓW:")
        print("-" * 40)
        
        best_alg = sorted_results[0]
        worst_alg = sorted_results[-1]
        
        print(f"🏆 Najlepszy: {best_alg[0]} ({best_alg[1]['mean_reward']:.3f})")
        print(f"📉 Najgorszy: {worst_alg[0]} ({worst_alg[1]['mean_reward']:.3f})")
        
        # Analiza korelacji parametrów z wynikami
        self._analyze_hyperparameter_impact()
    
    def _analyze_hyperparameter_impact(self):
        """Analizuje wpływ hiperparametrów na wyniki"""
        print("\n🔍 ANALIZA WPŁYWU HIPERPARAMETRÓW:")
        print("-" * 40)
        
        # Zbierz wszystkie hiperparametry
        all_params = set()
        for results in self.results.values():
            all_params.update(results['hyperparams'].keys())
        
        # Analizuj każdy parametr
        for param in ['learning_rate', 'gamma', 'batch_size']:
            if param in all_params:
                print(f"\n📈 {param}:")
                param_analysis = []
                
                for name, results in self.results.items():
                    if param in results['hyperparams']:
                        value = results['hyperparams'][param]
                        reward = results['mean_reward']
                        param_analysis.append((name, value, reward))
                
                # Sortuj według wartości parametru
                param_analysis.sort(key=lambda x: x[1])
                
                for name, value, reward in param_analysis:
                    print(f"   {name:<20} {value:<10} → {reward:.3f}")
    
    def evaluate_model_performance(self, algorithm_name: str, n_episodes: int = 50):
        """
        Szczegółowa ewaluacja konkretnego modelu
        
        Args:
            algorithm_name: Nazwa algorytmu do ewaluacji
            n_episodes: Liczba epizodów do ewaluacji
        """
        if algorithm_name not in self.results:
            print(f"❌ Algorytm {algorithm_name} nie został jeszcze wytrenowany")
            return
        
        print(f"\n🎯 SZCZEGÓŁOWA EWALUACJA: {algorithm_name}")
        print("-" * 50)
        
        model = self.results[algorithm_name]['model']
        eval_env = DummyVecEnv([make_env(self.n_players, seed=456)])
        
        # Ewaluacja z różnymi parametrami
        deterministic_rewards = []
        stochastic_rewards = []
        
        # Tryb deterministyczny
        for _ in range(n_episodes // 2):
            obs = eval_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                total_reward += reward[0]
            
            deterministic_rewards.append(total_reward)
        
        # Tryb stochastyczny
        for _ in range(n_episodes // 2):
            obs = eval_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, _ = eval_env.step(action)
                total_reward += reward[0]
            
            stochastic_rewards.append(total_reward)
        
        # Analiza wyników
        det_mean = np.mean(deterministic_rewards)
        det_std = np.std(deterministic_rewards)
        sto_mean = np.mean(stochastic_rewards)
        sto_std = np.std(stochastic_rewards)
        
        print(f"Deterministyczny: {det_mean:.3f} ± {det_std:.3f}")
        print(f"Stochastyczny:    {sto_mean:.3f} ± {sto_std:.3f}")
        print(f"Różnica:          {abs(det_mean - sto_mean):.3f}")
        
        eval_env.close()
    
    def save_results(self, filepath: str = "results/experiment_results.pkl"):
        """
        Zapisuje wyniki eksperymentu
        
        Args:
            filepath: Ścieżka do pliku wyników
        """
        try:
            # Przygotuj dane do zapisania (bez modeli - za duże)
            save_data = {}
            for name, results in self.results.items():
                save_data[name] = {
                    'mean_reward': results['mean_reward'],
                    'std_reward': results['std_reward'],
                    'training_time': results['training_time'],
                    'hyperparams': results['hyperparams'],
                    'total_timesteps': results['total_timesteps'],
                    'n_players': results['n_players'],
                    'final_model_path': results['final_model_path']
                }
            
            # Dodaj metadane
            save_data['_metadata'] = {
                'experiment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.device,
                'eval_freq': self.eval_freq,
                'n_eval_episodes': self.n_eval_episodes
            }
            
            # Zapisz do pliku
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"💾 Wyniki zapisane do: {filepath}")
            
        except Exception as e:
            print(f"❌ Błąd podczas zapisywania wyników: {e}")
    
    def load_results(self, filepath: str = "results/experiment_results.pkl"):
        """
        Ładuje wyniki eksperymentu
        
        Args:
            filepath: Ścieżka do pliku wyników
        """
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Usuń metadane
            if '_metadata' in loaded_data:
                metadata = loaded_data.pop('_metadata')
                print(f"📅 Dane z: {metadata.get('experiment_date', 'nieznana data')}")
            
            # Załaduj wyniki (bez modeli)
            for name, results in loaded_data.items():
                if name not in self.results:
                    self.results[name] = results
            
            print(f"📂 Wyniki załadowane z: {filepath}")
            print(f"   Algorytmy: {list(loaded_data.keys())}")
            
        except Exception as e:
            print(f"❌ Błąd podczas ładowania wyników: {e}")
    
    def get_best_algorithm(self) -> Optional[str]:
        """
        Zwraca nazwę najlepszego algorytmu
        
        Returns:
            Nazwa najlepszego algorytmu lub None
        """
        if not self.results:
            return None
        
        return max(self.results.items(), key=lambda x: x[1]['mean_reward'])[0]

# Test modułu
if __name__ == "__main__":
    print("🧪 Test modułu treningu...")
    
    # Test tworzenia trainera
    trainer = MultiAgentTrainer(n_players=4, verbose=2)
    print("✅ Trainer utworzony pomyślnie")
    
    # Test z prostą konfiguracją (tylko do testów)
    from stable_baselines3 import A2C
    
    simple_config = {
        "learning_rate": 1e-3,
        "n_steps": 5,
        "gamma": 0.99
    }
    
    try:
        # Krótki test treningu
        model = trainer.train_algorithm(
            "Test_A2C",
            A2C,
            simple_config,
            total_timesteps=1000  # Bardzo krótki test
        )
        print("✅ Test treningu zakończony pomyślnie")
        
        # Test porównania
        trainer.compare_algorithms()
        
        # Test zapisywania
        trainer.save_results("test_results.pkl")
        
    except Exception as e:
        print(f"⚠️  Test treningu pominięty: {e}")
    
    print("✅ Test modułu zakończony")
