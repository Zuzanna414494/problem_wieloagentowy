"""
Moduł do wizualizacji wyników eksperymentów wieloagentowych
w środowisku Texas Hold'em
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsVisualizer:
    """Klasa do tworzenia wizualizacji wyników eksperymentów"""
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """
        Inicjalizuje wizualizator
        
        Args:
            results: Słownik z wynikami eksperymentów
        """
        self.results = results
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Konfiguracja kolorów
        self.colors = {
            'PPO_Conservative': '#1f77b4',
            'PPO_Aggressive': '#ff7f0e', 
            'A2C_Standard': '#2ca02c',
            'DQN_Standard': '#d62728'
        }
        
        print(f"📊 Wizualizator zainicjalizowany dla {len(results)} algorytmów")
    
    def create_all_plots(self):
        """Tworzy wszystkie wykresy"""
        print("🎨 Tworzenie wizualizacji...")
        
        if not self.results:
            print("❌ Brak danych do wizualizacji")
            return
        
        # Podstawowe wykresy
        self.plot_algorithm_comparison()
        self.plot_performance_metrics()
        self.plot_hyperparameter_analysis()
        self.plot_training_efficiency()
        
        # Szczegółowe analizy
        self.plot_reward_distribution()
        self.plot_stability_analysis()
        self.create_summary_dashboard()
        
        print(f"✅ Wszystkie wykresy zapisane w folderze: {self.output_dir}")
    
    def plot_algorithm_comparison(self):
        """Wykres porównania algorytmów"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Przygotowanie danych
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[alg]['mean_reward'] for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        colors = [self.colors.get(alg, '#333333') for alg in algorithms]
        
        # Wykres 1: Średnie nagrody z błędami
        bars1 = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, 
                       capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Porównanie średnich nagród algorytmów', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Średnia nagroda')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Dodanie wartości na wykresie
        for bar, mean_val, std_val in zip(bars1, mean_rewards, std_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_val,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Wykres 2: Czas treningu
        training_times = [self.results[alg]['training_time']/60 for alg in algorithms]
        bars2 = ax2.bar(algorithms, training_times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Czas treningu algorytmów', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Czas treningu [minuty]')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Dodanie wartości na wykresie
        for bar, time_val in zip(bars2, training_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Wykres porównania algorytmów utworzony")
    
    def plot_performance_metrics(self):
        """Wykres metryk wydajności"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Przygotowanie danych
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[alg]['mean_reward'] for alg in algorithms]
        training_times = [self.results[alg]['training_time']/60 for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        
        # Wykres scatter z rozmiarem punktów proporcjonalnym do stabilności
        scatter = ax.scatter(training_times, mean_rewards, 
                           s=[300/(std if std > 1e-5 else 1e-5) for std in std_rewards],
                           c=[self.colors.get(alg, '#333333') for alg in algorithms],
                           alpha=0.7, edgecolors='black', linewidths=2)
        
        # Etykiety punktów
        for i, alg in enumerate(algorithms):
            ax.annotate(alg, (training_times[i], mean_rewards[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Czas treningu [minuty]', fontsize=12)
        ax.set_ylabel('Średnia nagroda', fontsize=12)
        ax.set_title('Wydajność vs Czas treningu\n(Rozmiar punktu = stabilność)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Dodanie linii trendu
        z = np.polyfit(training_times, mean_rewards, 1)
        p = np.poly1d(z)
        ax.plot(training_times, p(training_times), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Wykres metryk wydajności utworzony")
    
    def plot_hyperparameter_analysis(self):
        """Analiza wpływu hiperparametrów"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Parametry do analizy
        params_to_analyze = ['learning_rate', 'gamma', 'batch_size', 'ent_coef']
        
        for i, param in enumerate(params_to_analyze):
            ax = axes[i]
            
            # Zbieranie danych
            param_values = []
            rewards = []
            algorithm_names = []
            
            for alg_name, results in self.results.items():
                if param in results['hyperparams']:
                    param_values.append(results['hyperparams'][param])
                    rewards.append(results['mean_reward'])
                    algorithm_names.append(alg_name)
            
            if param_values:
                # Wykres scatter
                colors = [self.colors.get(alg, '#333333') for alg in algorithm_names]
                scatter = ax.scatter(param_values, rewards, c=colors, s=100, 
                                   alpha=0.7, edgecolors='black')
                
                # Etykiety
                for j, alg in enumerate(algorithm_names):
                    ax.annotate(alg, (param_values[j], rewards[j]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)
                
                ax.set_xlabel(param.replace('_', ' ').title())
                ax.set_ylabel('Średnia nagroda')
                ax.set_title(f'Wpływ {param} na wydajność')
                ax.grid(True, alpha=0.3)
                
                # Linia trendu jeśli możliwa
                if len(param_values) > 1:
                    try:
                        z = np.polyfit(param_values, rewards, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(param_values), max(param_values), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5)
                    except:
                        pass
            else:
                ax.text(0.5, 0.5, f'Brak danych\ndla {param}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Analiza hiperparametrów utworzona")
    
    def plot_training_efficiency(self):
        """Wykres efektywności treningu"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Przygotowanie danych
        algorithms = list(self.results.keys())
        efficiencies = []
        
        for alg in algorithms:
            reward = self.results[alg]['mean_reward']
            time = self.results[alg]['training_time'] / 60  # w minutach
            efficiency = reward / time if time > 0 else 0
            efficiencies.append(efficiency)
        
        # Wykres słupkowy z gradientem
        bars = ax.bar(algorithms, efficiencies, 
                     color=[self.colors.get(alg, '#333333') for alg in algorithms],
                     alpha=0.7, edgecolor='black', linewidth=2)
        
        # Dodanie wartości na słupkach
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Efektywność treningu algorytmów\n(Nagroda / Czas treningu)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Efektywność [nagroda/minuta]')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Dodanie średniej linii
        mean_efficiency = np.mean(efficiencies)
        ax.axhline(y=mean_efficiency, color='red', linestyle='--', 
                  label=f'Średnia: {mean_efficiency:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Wykres efektywności treningu utworzony")
    
    def plot_reward_distribution(self):
        """Rozkład nagród dla każdego algorytmu"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        algorithms = list(self.results.keys())
        
        for i, alg in enumerate(algorithms):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Symulacja rozkładu nagród na podstawie średniej i odchylenia
            mean_reward = self.results[alg]['mean_reward']
            std_reward = self.results[alg]['std_reward']
            
            # Generuj próbki rozkładu normalnego
            samples = np.random.normal(mean_reward, std_reward, 1000)
            
            # Histogram
            ax.hist(samples, bins=30, alpha=0.7, color=self.colors.get(alg, '#333333'),
                   edgecolor='black', density=True)
            
            # Krzywa rozkładu normalnego
            x = np.linspace(samples.min(), samples.max(), 100)
            y = (1 / (std_reward * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_reward) / std_reward) ** 2)
            ax.plot(x, y, 'r-', linewidth=2, label='Rozkład normalny')
            
            ax.set_title(f'{alg}\nμ={mean_reward:.3f}, σ={std_reward:.3f}')
            ax.set_xlabel('Nagroda')
            ax.set_ylabel('Gęstość')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Ukryj puste subplot'y
        for i in range(len(algorithms), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Rozkład nagród utworzony")
    
    def plot_stability_analysis(self):
        """Analiza stabilności algorytmów"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[alg]['mean_reward'] for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        colors = [self.colors.get(alg, '#333333') for alg in algorithms]
        
        # Wykres 1: Współczynnik zmienności
        cv_values = [std/abs(mean) if mean != 0 else 0 for mean, std in zip(mean_rewards, std_rewards)]
        
        bars1 = ax1.bar(algorithms, cv_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Stabilność algorytmów\n(Współczynnik zmienności)', fontweight='bold')
        ax1.set_ylabel('Współczynnik zmienności (σ/μ)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Dodanie wartości
        for bar, cv in zip(bars1, cv_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Wykres 2: Stosunek nagrody do wariancji
        reward_to_var = [mean/std if std > 0 else 0 for mean, std in zip(mean_rewards, std_rewards)]
        
        bars2 = ax2.bar(algorithms, reward_to_var, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Stosunek nagrody do wariancji\n(Wyższe = lepsze)', fontweight='bold')
        ax2.set_ylabel('Nagroda / Odchylenie')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Dodanie wartości
        for bar, ratio in zip(bars2, reward_to_var):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Analiza stabilności utworzona")
    
    def create_summary_dashboard(self):
        """Tworzy dashboard z podsumowaniem"""
        fig = plt.figure(figsize=(20, 12))
        
        # Definicja siatki
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Porównanie nagród
        ax1 = fig.add_subplot(gs[0, :2])
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[alg]['mean_reward'] for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        colors = [self.colors.get(alg, '#333333') for alg in algorithms]
        
        bars = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, 
                      capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Średnie nagrody algorytmów', fontweight='bold')
        ax1.set_ylabel('Nagroda')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Czas treningu
        ax2 = fig.add_subplot(gs[0, 2:])
        training_times = [self.results[alg]['training_time']/60 for alg in algorithms]
        ax2.pie(training_times, labels=algorithms, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax2.set_title('Proporcje czasu treningu', fontweight='bold')
        
        # 3. Scatter plot wydajności
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.scatter(training_times, mean_rewards, 
                   s=[300/(std if std > 1e-5 else 1e-5) for std in std_rewards],
                   c=colors, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Czas treningu [min]')
        ax3.set_ylabel('Średnia nagroda')
        ax3.set_title('Wydajność vs Czas', fontweight='bold')
        
        # 4. Tabela wyników
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.axis('off')
        
        # Przygotowanie danych do tabeli
        table_data = []
        for alg in algorithms:
            data = self.results[alg]
            table_data.append([
                alg,
                f"{data['mean_reward']:.3f}",
                f"{data['std_reward']:.3f}",
                f"{data['training_time']/60:.1f}m"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Algorytm', 'Średnia', 'Odch. std', 'Czas'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Tabela wyników', fontweight='bold', pad=20)
        
        # 5. Ranking algorytmów
        ax5 = fig.add_subplot(gs[2, :])
        
        # Sortowanie według nagrody
        sorted_algs = sorted(zip(algorithms, mean_rewards, std_rewards, training_times),
                           key=lambda x: x[1], reverse=True)
        
        y_pos = np.arange(len(sorted_algs))
        rewards = [x[1] for x in sorted_algs]
        alg_names = [x[0] for x in sorted_algs]
        
        bars = ax5.barh(y_pos, rewards, color=[self.colors.get(alg, '#333333') for alg in alg_names])
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(alg_names)
        ax5.set_xlabel('Średnia nagroda')
        ax5.set_title('Ranking algorytmów', fontweight='bold')
        
        # Dodanie medali
        medals = ['🥇', '🥈', '🥉', '4️⃣']
        for i, (bar, medal) in enumerate(zip(bars, medals)):
            if i < len(medals):
                ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        medal, ha='left', va='center', fontsize=16)
        
        # Główny tytuł
        fig.suptitle('Dashboard wyników eksperymentu Texas Hold\'em', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Dashboard podsumowujący utworzony")
    
    def generate_report_data(self) -> Dict[str, Any]:
        """Generuje dane do raportu"""
        if not self.results:
            return {}
        
        # Najlepszy algorytm
        best_alg = max(self.results.items(), key=lambda x: x[1]['mean_reward'])
        worst_alg = min(self.results.items(), key=lambda x: x[1]['mean_reward'])
        
        # Najszybszy trening
        fastest_alg = min(self.results.items(), key=lambda x: x[1]['training_time'])
        
        # Najstabilniejszy (najmniejsze std)
        most_stable = min(self.results.items(), key=lambda x: x[1]['std_reward'])
        
        # Statystyki ogólne
        all_rewards = [data['mean_reward'] for data in self.results.values()]
        all_times = [data['training_time'] for data in self.results.values()]
        
        report_data = {
            'experiment_summary': {
                'total_algorithms': len(self.results),
                'best_algorithm': best_alg[0],
                'best_score': best_alg[1]['mean_reward'],
                'worst_algorithm': worst_alg[0],
                'worst_score': worst_alg[1]['mean_reward'],
                'score_range': best_alg[1]['mean_reward'] - worst_alg[1]['mean_reward'],
                'fastest_algorithm': fastest_alg[0],
                'fastest_time': fastest_alg[1]['training_time'],
                'most_stable_algorithm': most_stable[0],
                'lowest_std': most_stable[1]['std_reward'],
                'average_score': np.mean(all_rewards),
                'average_time': np.mean(all_times),
                'total_training_time': sum(all_times)
            },
            'detailed_results': self.results
        }
        
        return report_data
    
    def save_report_data(self, filepath: str = None):
        """Zapisuje dane raportu do pliku"""
        if filepath is None:
            filepath = 'C:/Users/zuzan/PycharmProjects/problem_wieloagentowy/report_data.txt'
        
        report_data = self.generate_report_data()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("RAPORT Z EKSPERYMENTU TEXAS HOLD'EM\n")
            f.write("=" * 50 + "\n\n")
            
            summary = report_data['experiment_summary']
            f.write("PODSUMOWANIE EKSPERYMENTU:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Liczba algorytmów: {summary['total_algorithms']}\n")
            f.write(f"Najlepszy algorytm: {summary['best_algorithm']} ({summary['best_score']:.3f})\n")
            f.write(f"Najgorszy algorytm: {summary['worst_algorithm']} ({summary['worst_score']:.3f})\n")
            f.write(f"Rozstęp wyników: {summary['score_range']:.3f}\n")
            f.write(f"Najszybszy trening: {summary['fastest_algorithm']} ({summary['fastest_time']:.1f}s)\n")
            f.write(f"Najstabilniejszy: {summary['most_stable_algorithm']} (σ={summary['lowest_std']:.3f})\n")
            f.write(f"Średni wynik: {summary['average_score']:.3f}\n")
            f.write(f"Średni czas treningu: {summary['average_time']:.1f}s\n")
            f.write(f"Całkowity czas: {summary['total_training_time']:.1f}s\n\n")
            
            f.write("SZCZEGÓŁOWE WYNIKI:\n")
            f.write("-" * 30 + "\n")
            for alg_name, data in self.results.items():
                f.write(f"\n{alg_name}:\n")
                f.write(f"  Średnia nagroda: {data['mean_reward']:.3f} ± {data['std_reward']:.3f}\n")
                f.write(f"  Czas treningu: {data['training_time']:.1f}s\n")
                f.write(f"  Efektywność: {data['mean_reward']/(data['training_time']/60):.4f} nagroda/min\n")
                f.write(f"  Timesteps: {data['total_timesteps']}\n")
        
        print(f"📝 Dane raportu zapisane do: {filepath}")

# Test modułu
if __name__ == "__main__":
    print("🧪 Test modułu wizualizacji...")
    
    # Przykładowe dane testowe
    test_results = {
        'PPO_Conservative': {
            'mean_reward': 0.15,
            'std_reward': 0.05,
            'training_time': 300,
            'hyperparams': {'learning_rate': 3e-4, 'gamma': 0.99, 'batch_size': 64},
            'total_timesteps': 100000
        },
        'PPO_Aggressive': {
            'mean_reward': 0.12,
            'std_reward': 0.08,
            'training_time': 250,
            'hyperparams': {'learning_rate': 1e-3, 'gamma': 0.95, 'batch_size': 128},
            'total_timesteps': 100000
        },
        'A2C_Standard': {
            'mean_reward': 0.08,
            'std_reward': 0.04,
            'training_time': 200,
            'hyperparams': {'learning_rate': 7e-4, 'gamma': 0.99, 'batch_size': 32},
            'total_timesteps': 100000
        }
    }
    
    # Test wizualizatora
    visualizer = ResultsVisualizer(test_results)
    
    try:
        visualizer.create_all_plots()
        visualizer.save_report_data()
        print("✅ Test wizualizacji zakończony pomyślnie")
    except Exception as e:
        print(f"❌ Błąd podczas testu: {e}")