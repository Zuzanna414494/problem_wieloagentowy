"""
Konfiguracja algorytmów uczenia ze wzmocnieniem
dla środowiska Texas Hold'em
"""

from stable_baselines3 import PPO, A2C, DQN
from typing import Dict, Any
import torch


class AlgorithmsConfig:
    """Klasa zawierająca konfiguracje wszystkich algorytmów"""
    
    @staticmethod
    def get_algorithms() -> Dict[str, Dict[str, Any]]:
        """
        Zwraca słownik z konfiguracjami algorytmów
        
        Returns:
            Dict z konfiguracjami algorytmów dla eksperymentu na 6 punktów
        """
        return {
            # PPO - Wersja konserwatywna (stabilna)
            "PPO_Conservative": {
                "class": PPO,
                "params": {
                    "learning_rate": 3e-4,      # Standardowy learning rate
                    "n_steps": 2048,            # Więcej kroków na update
                    "batch_size": 64,           # Standardowy batch size
                    "n_epochs": 10,             # Więcej epok uczenia
                    "gamma": 0.99,              # Wysoki discount factor
                    "clip_range": 0.2,          # Konserwatywny clip range
                    "ent_coef": 0.01,          # Niska entropia dla stabilności
                    "vf_coef": 0.5,            # Standardowy współczynnik value function
                    "max_grad_norm": 0.5,      # Gradient clipping
                    "use_sde": False,          # Bez stochastic policy
                    "policy_kwargs": {
                        "net_arch": [32, 32],
                        "activation_fn": torch.nn.Tanh
                    }
                },
                "description": "PPO z konserwatywną konfiguracją dla stabilnego uczenia"
            },
            
            # PPO - Wersja agresywna (szybka)
            "PPO_Aggressive": {
                "class": PPO,
                "params": {
                    "learning_rate": 3e-4,
                    "n_steps": 1024,            # Mniej kroków na update
                    "batch_size": 128,          # Większy batch size
                    "n_epochs": 5,              # Mniej epok uczenia
                    "gamma": 0.95,              # Niższy discount factor
                    "clip_range": 0.3,          # Bardziej agresywny clip range
                    "ent_coef": 0.05,          # Wyższa entropia dla eksploracji
                    "vf_coef": 0.25,           # Niższy współczynnik value function
                    "max_grad_norm": 1.0,      # Mniej restrykcyjny gradient clipping
                    "use_sde": False,
                    "policy_kwargs": {
                        "net_arch": [32, 32],
                        "activation_fn": torch.nn.ReLU
                    }
                },
                "description": "PPO z agresywną konfiguracją dla szybkiego uczenia"
            },
            
            # A2C - Standardowa konfiguracja
            "A2C_Standard": {
                "class": A2C,
                "params": {
                    "learning_rate": 7e-4,      # Standardowy learning rate dla A2C
                    "n_steps": 5,               # Krótkie epizody
                    "gamma": 0.99,              # Standardowy discount factor
                    "ent_coef": 0.01,          # Entropia
                    "vf_coef": 0.25,           # Współczynnik value function
                    "max_grad_norm": 0.5,      # Gradient clipping
                    "rms_prop_eps": 1e-5,      # RMSprop epsilon
                    "policy_kwargs": {
                        "net_arch": [64, 64],   # Standardowa architektura
                        "activation_fn": torch.nn.Tanh
                    }
                },
                "description": "A2C ze standardową konfiguracją"
            },
            
            # DQN - Deep Q-Network (dla porównania)
            "DQN_Standard": {
                "class": DQN,
                "params": {
                    "learning_rate": 3e-4,
                    "buffer_size": 50000,       # Rozmiar replay buffer
                    "learning_starts": 1000,    # Kiedy zacząć uczenie
                    "batch_size": 32,           # Batch size
                    "tau": 1.0,                 # Target network update rate
                    "gamma": 0.99,              # Discount factor
                    "train_freq": 4,            # Częstotliwość treningu
                    "gradient_steps": 1,        # Kroki gradientu na update
                    "target_update_interval": 1000,  # Update target network
                    "exploration_fraction": 0.3, # Część czasu na eksplorację
                    "exploration_initial_eps": 1.0,  # Początkowy epsilon
                    "exploration_final_eps": 0.05,   # Końcowy epsilon
                    "policy_kwargs": {
                        "net_arch": [128, 128], # Architektura sieci
                        "activation_fn": torch.nn.ReLU
                    }
                },
                "description": "Deep Q-Network z experience replay"
            }
        }
    
    @staticmethod
    def get_algorithm_info(algorithm_name: str) -> Dict[str, Any]:
        """
        Zwraca informacje o konkretnym algorytmie
        
        Args:
            algorithm_name: Nazwa algorytmu
            
        Returns:
            Słownik z informacjami o algorytmie
        """
        algorithms = AlgorithmsConfig.get_algorithms()
        return algorithms.get(algorithm_name, {})
    
    @staticmethod
    def get_algorithm_names() -> list:
        """
        Zwraca listę nazw dostępnych algorytmów
        
        Returns:
            Lista nazw algorytmów
        """
        return list(AlgorithmsConfig.get_algorithms().keys())
    
    @staticmethod
    def print_algorithms_summary():
        """Wypisuje podsumowanie wszystkich algorytmów"""
        algorithms = AlgorithmsConfig.get_algorithms()
        
        print("📋 DOSTĘPNE ALGORYTMY:")
        print("=" * 50)
        
        for name, config in algorithms.items():
            print(f"\n🤖 {name}")
            print(f"   Klasa: {config['class'].__name__}")
            print(f"   Opis: {config['description']}")
            
            # Kluczowe parametry
            key_params = ['learning_rate', 'batch_size', 'gamma']
            print("   Kluczowe parametry:")
            for param in key_params:
                if param in config['params']:
                    print(f"     {param}: {config['params'][param]}")
        
        print("\n" + "=" * 50)
        
    @staticmethod
    def compare_hyperparameters():
        """Porównuje hiperparametry między algorytmami"""
        algorithms = AlgorithmsConfig.get_algorithms()
        
        print("📊 PORÓWNANIE HIPERPARAMETRÓW:")
        print("=" * 60)
        
        # Wspólne parametry do porównania
        common_params = ['learning_rate', 'gamma', 'batch_size']
        
        for param in common_params:
            print(f"\n🔧 {param.upper()}:")
            for name, config in algorithms.items():
                if param in config['params']:
                    value = config['params'][param]
                    print(f"   {name:20} : {value}")
        
        print("\n" + "=" * 60)

# Test konfiguracji
if __name__ == "__main__":
    print("🧪 Test konfiguracji algorytmów...")
    
    # Wypisz podsumowanie algorytmów
    AlgorithmsConfig.print_algorithms_summary()
    
    # Porównaj hiperparametry
    AlgorithmsConfig.compare_hyperparameters()
    
    # Test pojedynczego algorytmu
    ppo_config = AlgorithmsConfig.get_algorithm_info("PPO_Conservative")
    if ppo_config:
        print(f"\n✅ Konfiguracja PPO_Conservative załadowana")
        print(f"   Parametry: {len(ppo_config['params'])} elementów")
    
    print("✅ Test konfiguracji zakończony pomyślnie")