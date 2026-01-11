import numpy as np
import matplotlib.pyplot as plt

class CargarCarril:
    def __init__(self, v_info, li_geom, tramos_con_carga, num_fle):
        self.v_info = v_info[1:-1, :]
        self.li_geom = li_geom
        self.tramos_con_carga = tramos_con_carga
        self.num_fle = num_fle
        self.long_tram = np.append(li_geom[:, 0], li_geom[-1, 1])
        self.nnodos = li_geom.shape[0] + 1

    def calcular_areas_tramos(self, step):
        for tramo in self.li_geom:
            tramo_indices = np.where((self.v_info[:, 0] >= tramo[0]) & (self.v_info[:, 0] <= tramo[1]))[0]
            tramo_info = self.v_info[tramo_indices, :]
            area_positiva = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] < 0])) * step
            area_negativa = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] > 0])) * step

    def graficar(self):
        plt.style.use('classic')
        fig, ax = plt.subplots()
        ax.plot(self.v_info[:, 0], self.v_info[:, 1])
        step = self.v_info[1, 0] - self.v_info[0, 0]
        self.calcular_areas_tramos(step)
        plt.fill_between(self.v_info[:, 0], self.v_info[:, 1], 0, where=(self.v_info[:, 1] < 0), color='lightcoral', alpha=0.3)
        plt.fill_between(self.v_info[:, 0], self.v_info[:, 1], 0, where=(self.v_info[:, 1] >= 0), color='skyblue', alpha=0.3)
        ax.set_xlabel('Distancia del puente')
        ax.set_ylabel('Valor en la línea de influencia')
        ax.set_title('Linea de influencia')
        x_coords = np.hstack((self.li_geom[:, 0], self.li_geom[-1, 1]))
        y_coords = np.zeros(self.nnodos)
        ax.plot(x_coords, y_coords, color='#000000', linewidth=1.5)
        max_index = np.argmax(np.abs(self.v_info[:, 1]))
        max_value = self.v_info[max_index, 1]
        max_x = self.v_info[max_index, 0]
        markers = ['s'] + ['^'] * (len(x_coords) - 1)
        colors = ['#000000'] * len(x_coords)
        for x, y, marker, color in zip(x_coords, y_coords, markers, colors):
            ax.scatter(x, y, color=color, marker=marker, s=60)
        for i, carga in enumerate(self.tramos_con_carga):
            if carga:
                x_tramo = np.linspace(self.long_tram[i], self.long_tram[i + 1], self.num_fle + 1)
                y_tramo = np.zeros_like(x_tramo) + 0.5
                for x, y in zip(x_tramo, y_tramo):
                    ax.arrow(x, y, 0, -0.5, head_width=1, head_length=0.1,
                             fc='blue', ec='blue', length_includes_head=True)     
                rect = plt.Rectangle((x_tramo[0] - 0.05 * 0.5, y_tramo[0]),
                                     (x_tramo[-1] - x_tramo[0]) + 0.05, -0.1,
                                     color='blue', alpha=0.5)
                ax.add_patch(rect)
                centro_tramo = (x_tramo[0] + x_tramo[-1]) / 2
                ax.annotate(f'q = {0.952} Tn/m', xy=(centro_tramo, 0), xytext=(centro_tramo, 0.7),
                            arrowprops=dict(facecolor='black', shrink=2),
                            fontsize=10, ha='center', weight='bold')
        ax.grid(True)
        return fig