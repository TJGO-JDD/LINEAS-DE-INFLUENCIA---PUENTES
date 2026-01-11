import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import ipywidgets as widgets

class LIMomento:
    def __init__(self, LI_Geom, Xcoord, Step):
        self.LI_Geom = LI_Geom
        self.Xcoord = Xcoord
        self.Step = Step
        self.Input1 = 'M'
        self.VInfo = self.calculate_VInfo()    
    def calculate_VInfo(self):
        Nnodos = self.LI_Geom.shape[0] + 1
        MRig = np.zeros((Nnodos, Nnodos))
        iterr = np.array([0, 1])
        
        for ke in range(Nnodos - 1):
            Lenght_e = self.LI_Geom[ke, 1] - self.LI_Geom[ke, 0]
            EI_e = self.LI_Geom[ke, 2] * self.LI_Geom[ke, 3]
            Kr_e = (2 * EI_e) / Lenght_e
            ke_e = np.array([[2 * Kr_e, Kr_e], [Kr_e, 2 * Kr_e]])
            for i in iterr:
                for j in iterr:
                    MRig[ke + i, ke + j] = ke_e[i, j] + MRig[ke + i, ke + j]

        Np = (self.LI_Geom[Nnodos - 2, 1] - self.LI_Geom[0, 0]) / self.Step
        VInfo = np.zeros((int(Np + 1), 2))

        for i in range(int(Np)):
            XLoad = i * self.Step + self.LI_Geom[0, 0]
            Forces = np.zeros([Nnodos, 1])
            VInfo[i, 0] = XLoad

            for p in range(Nnodos - 1):
                if self.LI_Geom[p, 0] < XLoad < self.LI_Geom[p, 1]:
                    L = self.LI_Geom[p, 1] - self.LI_Geom[p, 0]
                    kp = (XLoad - self.LI_Geom[p, 0]) / L
                    Forces[p, 0] = -L * kp * (1 - kp) ** 2
                    Forces[p + 1, 0] = L * (kp ** 2) * (1 - kp)

            Theta = np.linalg.solve(MRig, Forces)

            for p in range(Nnodos - 1):
                if self.LI_Geom[p, 0] < self.Xcoord < self.LI_Geom[p, 1]:
                    L = self.LI_Geom[p, 1] - self.LI_Geom[p, 0]
                    EI = self.LI_Geom[p, 2] * self.LI_Geom[p, 3]
                    Kr = 2 * EI / L
                    Fe = np.zeros((Nnodos + 1, 1))
                    if self.LI_Geom[p, 0] < XLoad < self.LI_Geom[p, 1]:
                        kp = (XLoad - self.LI_Geom[p, 0]) / L
                        Fe[p, 0] = -L * kp * (1 - kp) ** 2
                        Fe[p + 1, 0] = L * (kp ** 2) * (1 - kp)
                    Ml = 2 * Theta[p, 0] * Kr + Theta[p + 1, 0] * Kr - Fe[p, 0]
                    Mr = Theta[p, 0] * Kr + 2 * Theta[p + 1, 0] * Kr - Fe[p + 1, 0]

                    if self.Input1 == 'M':
                        if abs(Fe[p, 0]) > 0 and abs(Fe[p + 1, 0]) > 0:
                            kp1 = (XLoad - self.LI_Geom[p, 0]) / L
                            kp2 = (self.Xcoord - self.LI_Geom[p, 0]) / L
                            Rl = ((Ml + Mr) / L) + (1 - kp1)
                            Rr = 1 - Rl
                            if kp2 < kp1:
                                VInfo[i, 1] = Rl * kp2 * L - Ml
                            else:
                                VInfo[i, 1] = Mr + Rr * (1 - kp2) * L
                        else:
                            Rl = (Ml + Mr) / L
                            kp2 = (self.Xcoord - self.LI_Geom[p, 0]) / L
                            VInfo[i, 1] = Rl * kp2 * L - Ml
                    else:
                        if abs(Fe[p, 0]) > 0 and abs(Fe[p + 1, 0]) > 0:
                            kp1 = (XLoad - self.LI_Geom[p, 0]) / L
                            kp2 = (self.Xcoord - self.LI_Geom[p, 0]) / L
                            Rl = ((Ml + Mr) / L) + (1 - kp1)
                            Rr = 1 - Rl
                            if kp2 < kp1:
                                VInfo[i, 1] = Rl
                            else:
                                VInfo[i, 1] = -Rr
                        else:
                            Rl = (Ml + Mr) / L
                            VInfo[i, 1] = Rl
        return VInfo[1:-1, :]
    
    def Atramos(self):
        Nnodos = self.LI_Geom.shape[0] + 1
        Atram = np.zeros((Nnodos-1,2))  
        i=0
        for tramo in self.LI_Geom:
            tramo_indices = np.where((self.VInfo[:, 0] >= tramo[0]) & (self.VInfo[:, 0] <= tramo[1]))[0]
            tramo_info = self.VInfo[tramo_indices, :]
            Atram[i,0] = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] < 0])) * self.Step
            Atram[i,1] = np.sum(np.abs(tramo_info[:, 1][tramo_info[:, 1] > 0])) * self.Step
            i=i+1
            #print(f"Tramo {tramo[0]}-{tramo[1]} -> Area +: {Atramos[i,0]}, Area -: {Atramos[i,1]}")
        return Atram

    def plot(self):
        plt.style.use('classic')
        fig, ax = plt.subplots()
        ax.plot(self.VInfo[:, 0], self.VInfo[:, 1])      

        AreaPositiva = np.sum(np.abs(self.VInfo[:, 1][self.VInfo[:, 1] < 0])) * self.Step
        AreaNegativa = np.sum(np.abs(self.VInfo[:, 1][self.VInfo[:, 1] > 0])) * self.Step
        plt.fill_between(self.VInfo[:, 0], self.VInfo[:, 1], 0, where=(self.VInfo[:, 1] < 0), color='lightcoral', alpha=0.3)
        plt.fill_between(self.VInfo[:, 0], self.VInfo[:, 1], 0, where=(self.VInfo[:, 1] >= 0), color='skyblue', alpha=0.3)

        ax.set_xlabel('Distancia del puente')
        ax.set_ylabel('Valor en la línea de influencia')
        ax.set_title('LINEA DE INFLUCENCIA DE MOMENTOS')

        x_coords = np.hstack((self.LI_Geom[:, 0], self.LI_Geom[-1, 1]))
        y_coords = np.zeros(len(x_coords))
        ax.plot(x_coords, y_coords, color='#000000', linewidth=1.5)

        Max = max(abs(self.VInfo[:, 1]))
        for fila in self.VInfo:
            if fila[1] / Max == 1:
                X_0 = fila[0]
                ax.plot(X_0, fila[1], 'ro', markersize=7)
                ax.plot([X_0, X_0], [fila[1], 0], color='red', linestyle='--')
                break
            elif fila[1] / Max == -1:
                X_0 = fila[0]
                ax.plot(X_0, -1 * Max, 'ro', markersize=7)
                ax.plot([X_0, X_0], [-1 * Max, 0], color='red', linestyle='--')
                break

        max_index = np.argmax(np.abs(self.VInfo[:, 1]))
        max_value = self.VInfo[max_index, 1]
        max_x = self.VInfo[max_index, 0]
        ax.annotate(f'Máx: {max_value:.3f}', (max_x, max_value), xytext=(max_x + 0.5, max_value),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))

        markers = ['s'] + ['^'] * (len(x_coords) - 1)
        colors = ['#000000'] * len(x_coords)
        for x, y, marker, color in zip(x_coords, y_coords, markers, colors):
            ax.scatter(x, y, color=color, marker=marker, s=60)

        ax.grid(True)
        return fig
        

    def MaxValue(self):
        return np.max(np.abs(self.VInfo[:, 1]))

    def AreaPositiva(self):
        return np.sum(self.VInfo[:, 1][self.VInfo[:, 1] > 0]) * self.Step

    def AreaNegativa(self):
        return np.sum(self.VInfo[:, 1][self.VInfo[:, 1] < 0]) * self.Step
    