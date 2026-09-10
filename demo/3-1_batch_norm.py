import numpy as np
import matplotlib.pyplot as plt

# Симулируем работу одного нейрона
np.random.seed(42)

# 1. Вход в BN (сырые активации с предыдущего слоя)
x = np.random.randn(1000) * 3 + 1  # Среднее ~1, Std ~3

# 2. Нормализация (обнуление среднего и единичная дисперсия)
x_norm = (x - np.mean(x)) / np.std(x)  # Теперь mean ≈ 0, std ≈ 1

# 3. Восстановление диапазона (обучаемые параметры)
gamma = 1.5   # Сеть решила увеличить дисперсию
beta = -0.5   # Сеть решила сместить влево
y = x_norm * gamma + beta

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(x, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title(f'1. До BN (raw)\nmean={np.mean(x):.2f}, std={np.std(x):.2f}')
axes[0].set_xlabel('Activation')

axes[1].hist(x_norm, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title(f'2. После нормализации\nmean={np.mean(x_norm):.2f}, std={np.std(x_norm):.2f}')
axes[1].set_xlabel('Normalized Activation')

axes[2].hist(y, bins=30, alpha=0.7, color='red', edgecolor='black')
axes[2].set_title(f'3. После восстановления (γ={gamma}, β={beta})\nmean={np.mean(y):.2f}, std={np.std(y):.2f}')
axes[2].set_xlabel('Restored Activation')

plt.tight_layout()
plt.show()
