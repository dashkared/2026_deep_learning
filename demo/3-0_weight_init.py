import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Настройка красивых графиков
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

def visualize_deep_network_activations(
    num_layers=10,
    layer_size=256,  # Количество нейронов в каждом скрытом слое
    batch_size=1000,
    weight_std=2.0,  # Большая дисперсия для демонстрации проблемы
    activation='relu'
):
    """
    Визуализация активаций в глубокой сети с плохой инициализацией
    
    Args:
        num_layers: количество скрытых слоев
        layer_size: размер каждого слоя
        batch_size: количество объектов в батче
        weight_std: стандартное отклонение весов (большое -> взрыв)
        activation: функция активации ('relu', 'tanh')
    """
    
    # 1. Входные данные: стандартное нормальное распределение
    X = np.random.randn(batch_size, layer_size)
    
    # Сохраняем активации для каждого слоя
    activations = {'input': X.copy()}
    current = X.copy()
    
    # 2. Проход по слоям
    for layer_idx in range(1, num_layers + 1):
        # Инициализация весов с БОЛЬШОЙ дисперсией
        W = np.random.randn(layer_size, layer_size) * weight_std
        b = np.zeros(layer_size)  # смещения нулевые
        
        # Линейное преобразование
        Z = current @ W + b
        
        # Применяем ReLU
        if activation == 'relu':
            A = np.maximum(0, Z)
        elif activation == 'tanh':
            A = np.tanh(Z)
        else:
            A = Z
        
        # Сохраняем активации
        activations[f'layer_{layer_idx}'] = A
        current = A
        
    return activations

def plot_activations_statistics(activations, num_layers=10):
    """
    Построение графиков статистики активаций
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Цветовая схема
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_layers + 1))
    
    # ----- 1. График средних и стандартных отклонений -----
    ax1 = plt.subplot(2, 3, 1)
    
    means = []
    stds = []
    layer_names = []
    
    for layer_name, data in activations.items():
        # Усредняем по батчу, оставляя размерность признаков
        mean_per_neuron = np.mean(data, axis=0)
        std_per_neuron = np.std(data, axis=0)
        
        # Общая статистика по всем нейронам
        means.append(np.mean(mean_per_neuron))
        stds.append(np.mean(std_per_neuron))
        layer_names.append(layer_name)
    
    # Строим линии
    x = np.arange(len(layer_names))
    ax1.plot(x, means, 'o-', color='blue', label='Mean', linewidth=2, markersize=8)
    ax1.fill_between(x, 
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.3, color='blue', label='Mean ± Std')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Activation Value')
    ax1.set_title('Mean and Std of Activations\n(бледная синяя область = std)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ----- 2. График доли "мертвых" нейронов (для ReLU) -----
    ax2 = plt.subplot(2, 3, 2)
    
    dead_neurons = []
    for layer_name, data in activations.items():
        if layer_name == 'input':
            continue
        # Считаем нейроны, которые всегда 0 для всех объектов в батче
        dead_mask = np.all(data == 0, axis=0)
        dead_ratio = np.mean(dead_mask) * 100
        dead_neurons.append(dead_ratio)
    
    layers = [f'Layer {i}' for i in range(1, len(dead_neurons) + 1)]
    bars = ax2.bar(layers, dead_neurons, color='coral', edgecolor='darkred')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Dead Neurons (%)')
    ax2.set_title(f'Proportion of "Dead" Neurons\n(Total: {dead_neurons[-1]:.1f}% in last layer)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, dead_neurons):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # ----- 3. Гистограмма активаций для каждого слоя -----
    ax3 = plt.subplot(2, 3, 3)
    
    # Берем первый, средний и последний слой
    layers_to_plot = [
        ('Input', activations['input']),
        ('Layer 1', activations['layer_1']),
        (f'Layer {num_layers//2}', activations[f'layer_{num_layers//2}']),
        (f'Layer {num_layers}', activations[f'layer_{num_layers}'])
    ]
    
    # Превращаем все активации в один массив для гистограммы
    for idx, (name, data) in enumerate(layers_to_plot):
        flat_data = data.flatten()
        ax3.hist(flat_data, bins=50, alpha=0.5, 
                label=f'{name} (mean={np.mean(flat_data):.2f}, std={np.std(flat_data):.2f})',
                density=True, histtype='step', linewidth=2,
                color=plt.cm.tab10(idx))
    
    ax3.set_xlabel('Activation Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Activations\n(сравнение разных слоев)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ----- 4. Heatmap активаций (первые 50 нейронов, 100 объектов) -----
    ax4 = plt.subplot(2, 3, 4)
    
    # Берем активации последнего слоя
    last_layer_data = activations[f'layer_{num_layers}']
    # Ограничиваем для читаемости
    sample_data = last_layer_data[:100, :50]
    
    im = ax4.imshow(sample_data, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
    ax4.set_xlabel('Neuron index')
    ax4.set_ylabel('Sample index')
    ax4.set_title(f'Layer {num_layers} Activations Heatmap\n(черный = 0 для ReLU)')
    plt.colorbar(im, ax=ax4)
    
    # ----- 5. Эволюция среднего и максимума по слоям -----
    ax5 = plt.subplot(2, 3, 5)
    
    max_vals = []
    mean_vals = []
    for layer_name, data in activations.items():
        max_vals.append(np.max(data))
        mean_vals.append(np.mean(data))
    
    ax5.plot(layer_names, max_vals, 'o-', color='red', label='Max', linewidth=2, markersize=8)
    ax5.plot(layer_names, mean_vals, 's-', color='blue', label='Mean', linewidth=2, markersize=8)
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Value')
    ax5.set_title('Max vs Mean Activation\n(показывает взрыв в максимуме)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layer_names, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # Логарифмическая шкала для наглядности взрыва
    
    # ----- 6. Дополнительно: распределение весов -----
    # Здесь мы показываем, что веса тоже взрываются
    ax6 = plt.subplot(2, 3, 6)
    
    # Симулируем несколько слоев весов
    for i in range(1, 6, 2):
        W = np.random.randn(1000) * (2.0 ** i)  # Дисперсия растет с каждым слоем
        ax6.hist(W, bins=50, alpha=0.4, 
                label=f'Layer {i} (std={np.std(W):.1f})',
                density=True, histtype='step', linewidth=2)
    
    ax6.set_xlabel('Weight Value')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution of Weights in Different Layers\n(дисперсия экспоненциально растет)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================
# ЗАПУСК ЭКСПЕРИМЕНТА
# ============================================

# Параметры
NUM_LAYERS = 10
LAYER_SIZE = 256
BATCH_SIZE = 1000
WEIGHT_STD = 2.0  # Критически большая дисперсия

print("=" * 60)
print("ЭКСПЕРИМЕНТ: Глубокая сеть с плохой инициализацией")
print(f"Слоев: {NUM_LAYERS}, Нейронов в слое: {LAYER_SIZE}")
print(f"Стандартное отклонение весов: {WEIGHT_STD}")
print("=" * 60)

# Запускаем эксперимент
activations = visualize_deep_network_activations(
    num_layers=NUM_LAYERS,
    layer_size=LAYER_SIZE,
    batch_size=BATCH_SIZE,
    weight_std=WEIGHT_STD,
    activation='relu'
)

# Визуализация
fig = plot_activations_statistics(activations, NUM_LAYERS)

# Дополнительная аналитика в консоль
print("\n📊 СТАТИСТИКА АКТИВАЦИЙ ПО СЛОЯМ:")
print("-" * 60)
print(f"{'Layer':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'% Dead':<10}")
print("-" * 60)

for layer_name, data in activations.items():
    if layer_name == 'input':
        continue
    flat_data = data.flatten()
    dead_ratio = np.mean(flat_data == 0) * 100
    print(f"{layer_name:<15} {np.mean(flat_data):<12.4f} {np.std(flat_data):<12.4f} "
          f"{np.min(flat_data):<12.4f} {np.max(flat_data):<12.4f} {dead_ratio:<10.1f}%")

print("-" * 60)
print("\n⚠️  ВНИМАНИЕ: Видите, как значения становятся огромными?")
print("   Это классический пример ВЗРЫВА ГРАДИЕНТОВ/АКТИВАЦИЙ!")
print("   После 5-го слоя ReLU выдает числа порядка десятков тысяч.")
print("   Градиенты тоже будут такими же огромными -> обучение нестабильно.")

plt.show()

# ============================================
# ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: Сравнение с хорошей инициализацией
# ============================================

def compare_init_methods():
    """Сравнение плохой (std=2) и хорошей (Xavier) инициализации"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Функция для получения статистики
    def get_stats(weight_std, activation='relu'):
        activ = visualize_deep_network_activations(
            num_layers=10, weight_std=weight_std, activation=activation
        )
        means = []
        stds = []
        for name, data in activ.items():
            if name == 'input':
                continue
            flat = data.flatten()
            means.append(np.mean(flat))
            stds.append(np.std(flat))
        return means, stds
    
    # Плохая инициализация
    means_bad, stds_bad = get_stats(2.0)
    
    # Хорошая инициализация (Xavier для ReLU: std = sqrt(2 / fan_in))
    xavier_std = np.sqrt(2 / LAYER_SIZE)  # ~0.088
    means_good, stds_good = get_stats(xavier_std)
    
    layers = list(range(1, 11))
    
    # График средних
    axes[0].plot(layers, means_bad, 'o-', color='red', label='Bad init (std=2)', linewidth=2)
    axes[0].plot(layers, means_good, 's-', color='green', label=f'Xavier init (std={xavier_std:.3f})', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Activation')
    axes[0].set_title('Mean of Activations\n(сравнение инициализаций)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График стандартных отклонений
    axes[1].plot(layers, stds_bad, 'o-', color='red', label='Bad init (std=2)', linewidth=2)
    axes[1].plot(layers, stds_good, 's-', color='green', label=f'Xavier init (std={xavier_std:.3f})', linewidth=2)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Std of Activations')
    axes[1].set_title('Standard Deviation of Activations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Boxplot для последнего слоя
    bad_last = np.random.randn(1000) * stds_bad[-1] + means_bad[-1]
    good_last = np.random.randn(1000) * stds_good[-1] + means_good[-1]
    
    axes[2].boxplot([bad_last, good_last], 
                     labels=['Bad Init\n(взрыв)', 'Xavier Init\n(стабильно)'],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
    axes[2].set_ylabel('Activation Value')
    axes[2].set_title('Distribution in Last Layer\n(сравнение на выходе)')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# Запускаем сравнение
fig_compare = compare_init_methods()
plt.show()
