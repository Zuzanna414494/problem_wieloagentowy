"""
Wrapper dla środowiska Texas Hold'em z PettingZoo
"""

import numpy as np
import gymnasium as gym
from pettingzoo.classic import texas_holdem_v4
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TexasHoldemWrapper(gym.Env):
    """Wrapper dla środowiska Texas Hold'em do treningu z Stable-Baselines3"""
    
    def __init__(self, n_players: int = 4, seed: Optional[int] = None):
        self.n_players = max(2, min(10, n_players))
        self.seed = seed
        self.env = None
        self.agents = None
        self.current_agent = None
        self._last_action_mask = None  
        self.reset_env()  # ustawia środowisko

    def reset_env(self):
        """Resetuje środowisko Texas Hold'em i zwraca pierwszą obserwację"""
        try:
            self.env = texas_holdem_v4.env(num_players=self.n_players, render_mode="human") 
            obs = self.env.reset(seed=self.seed) if self.seed is not None else self.env.reset()
            self.agents = self.env.agents.copy()
            self.current_agent = None
            obs, _, _, _, _ = self.env.last()
            return obs
        except Exception as e:
            print(f"Błąd podczas resetowania środowiska: {e}")
            raise


    def get_action_mask(self):
        """Zwraca maskę legalnych akcji"""
        return self._last_action_mask if self._last_action_mask is not None else np.ones(self.action_space.n, dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        obs = self.reset_env()
        return self._process_observation(obs), {}

    def _step_internal(self, action):
        """Wykonuje pojedynczy krok w środowisku"""
        self.env.step(action)
        obs, reward, done, _, info = self.env.last()
        return obs, reward, done, info

    def step(self, action):
        if self._last_action_mask is not None:
            legal_actions = np.where(self._last_action_mask == 1)[0].tolist()
            if action not in legal_actions:
                action = np.random.choice(legal_actions)

        obs, reward, done, info = self._step_internal(action)

        # 🔍 DODAJ renderowanie co pewien czas:
        if hasattr(self, "_step_counter"):
            self._step_counter += 1
        else:
            self._step_counter = 1

        if self._step_counter % 100 == 0:  # co 100 kroków
            try:
                self.env.render()
            except:
                pass  # w razie błędu nic nie rób

        return self._process_observation(obs), reward, done, False, info

    def _process_observation(self, obs_dict):
        """Konwertuje złożone dict-y obserwacji na wektor NumPy (float32)"""
        obs_vector = []

        for key, value in obs_dict.items():
            if isinstance(value, (int, float, bool)):
                obs_vector.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                obs_vector.extend([float(v) for v in value])
            elif isinstance(value, dict):
                for subvalue in value.values():
                    if isinstance(subvalue, (int, float, bool)):
                        obs_vector.append(float(subvalue))
                    elif isinstance(subvalue, (list, np.ndarray)):
                        obs_vector.extend([float(v) for v in subvalue])
        
        if "action_mask" in obs_dict:
            self._last_action_mask = np.array(obs_dict["action_mask"], dtype=np.float32)

        # Sanity check
        obs_array = np.array(obs_vector, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
        return obs_array

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(76,),  # ✅ TYLE JEST W OBECNEJ OBSERWACJI!
            dtype=np.float32
        )

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(4)

    def get_legal_actions(self) -> list:
        try:
            if self.env and hasattr(self.env, 'last'):
                observation, _, _, _, info = self.env.last()
                if isinstance(observation, dict) and 'legal_actions' in observation:
                    return observation['legal_actions']
        except:
            pass
        return list(range(self.action_space.n))

    def close(self):
        if self.env:
            try:
                self.env.close()
            except:
                pass

    def __del__(self):
        self.close()


def make_env(n_players: int = 4, seed: Optional[int] = None):
    def _init():
        try:
            env = TexasHoldemWrapper(n_players=n_players, seed=seed)
            env = Monitor(env, allow_early_resets=True)
            return env
        except Exception as e:
            print(f"Błąd podczas tworzenia środowiska: {e}")
            raise
    return _init


# Test środowiska
if __name__ == "__main__":
    print("🧪 Test środowiska Texas Hold'em...")

    try:
        env = TexasHoldemWrapper(n_players=4)
        print(f"✅ Środowisko utworzone pomyślnie")
        print(f"   Przestrzeń obserwacji: {env.observation_space}")
        print(f"   Przestrzeń akcji: {env.action_space}")

        obs, _ = env.reset()
        print(f"✅ Reset pomyślny, obserwacja: {obs.shape}")

        legal_actions = env.get_legal_actions()
        action = legal_actions[0] if legal_actions else 0
        obs, reward, done, _, _ = env.step(action)
        print(f"✅ Krok pomyślny, nagroda: {reward}, zakończono: {done}")

        env.close()
        print("✅ Test zakończony pomyślnie")

    except Exception as e:
        print(f"❌ Błąd podczas testu: {e}")
