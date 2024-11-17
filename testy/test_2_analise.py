import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dane pomiarowe
serie1 = {
    "300 mm": [308, 278, 298, 292, 313, 295, 295, 301],
    "500 mm": [515, 494, 479, 484, 485, 490, 485, 488],
    "700 mm": [692, 689, 693, 693, 667, 704, 686, 691],
    "1000 mm": [979, 985, 976, 976, 987, 976, 989, 990]
}


# Tworzenie DataFrame
df1 = pd.DataFrame(serie1)

# Obliczanie średniej i odchylenia standardowego
mean1 = df1.mean()
std1 = df1.std()


# Wyświetlanie wyników
print("\nSeria 1 - Średnie wartości:")
print(mean1)
print("\nSeria 1 - Odchylenie standardowe:")
print(std1)


# Wizualizacja danych
plt.figure(figsize=(10, 6))

# Wykres punktowy
plt.plot(df1, marker='o', linestyle='')
plt.title('Pomiar odległości kamerą RealSense D435f')
plt.xlabel('Numer pomiaru')
plt.ylabel('Odległość [mm]')



plt.tight_layout()
plt.show()
