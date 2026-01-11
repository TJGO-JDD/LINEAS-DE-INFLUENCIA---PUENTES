import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class LICortante:
    def __init__(self, LI_Geom, Xcoord, Step):
        self.LI_Geom = LI_Geom
        self.Xcoord = Xcoord
        self.Step = Step
        self.Nnodos = LI_Geom.shape[0] + 1
        self.MRig = np.zeros((self.Nnodos, self.Nnodos))
        self.iterr = np.array([0, 1])
        self.VInfo = None
        self.__setup_rigidity_matrix()
        self.__calculate_vinfo()

    def __setup_rigidity_matrix(self):
        for ke in range(self.Nnodos - 1):
            Lenght_e = self.LI_Geom[ke, 1] - self.LI_Geom[ke, 0]
            EI_e = self.LI_Geom[ke, 2] * self.LI_Geom[ke, 3]
            Kr_e = (2 * EI_e) / Lenght_e
            ke_e = np.array([[2 * Kr_e, Kr_e],
                             [Kr_e, 2 * Kr_e]])
            for i in self.iterr:
                for j in self.iterr:
                    self.MRig[ke + i, ke + j] = ke_e[i, j] + self.MRig[ke + i, ke + j]

    def __calculate_vinfo(self):
        Np = (self.LI_Geom[self.Nnodos - 2, 1] - self.LI_Geom[0, 0]) / self.Step
        self.VInfo = np.zeros((int(Np + 1), 2))
        
        for i in range(int(Np)):
            XLoad = i * self.Step + self.LI_Geom[0, 0]
            Forces = np.zeros([self.Nnodos, 1])
            self.VInfo[i, 0] = XLoad
            
            for p in range(self.Nnodos - 1):
                if XLoad > self.LI_Geom[p, 0] and XLoad < self.LI_Geom[p, 1]:
                    L = self.LI_Geom[p, 1] - self.LI_Geom[p, 0]
                    kp = (XLoad - self.LI_Geom[p, 0]) / L
                    Forces[p, 0] = -L * kp * (1 - kp)**2
                    Forces[p + 1, 0] = L * (kp**2) * (1 - kp)
            
            Theta = np.linalg.solve(self.MRig, Forces)
            
            for p in range(self.Nnodos - 1):
                if self.Xcoord > self.LI_Geom[p, 0] and self.Xcoord < self.LI_Geom[p, 1]:
                    L = self.LI_Geom[p, 1] - self.LI_Geom[p, 0]
                    EI = self.LI_Geom[p, 2] * self.LI_Geom[p, 3]
                    Kr = 2 * EI / L
                    Fe = np.zeros((self.Nnodos + 1, 1))
                    if XLoad > self.LI_Geom[p, 0] and XLoad < self.LI_Geom[p, 1]:
                        kp = (XLoad - self.LI_Geom[p, 0]) / L
                        Fe[p, 0] = -L * kp * (1 - kp)**2
                        Fe[p + 1, 0] = L * (kp**2) * (1 - kp)
                    else:
                        Fe[p, 0] = 0
                        Fe[p + 1, 0] = 0
                    Ml = 2 * Theta[p, 0] * Kr + Theta[p + 1, 0] * Kr - Fe[p, 0]
                    Mr = Theta[p, 0] * Kr + 2 * Theta[p + 1, 0] * Kr - Fe[p + 1, 0]
                    
                    if abs(Fe[p, 0]) > 0 and abs(Fe[p + 1, 0]) > 0:
                        kp1 = (XLoad - self.LI_Geom[p, 0]) / L
                        kp2 = (self.Xcoord - self.LI_Geom[p, 0]) / L
                        Rl = ((Ml + Mr) / L) + (1 - kp1)
                        Rr = 1 - Rl
                        if kp2 < kp1:
                            self.VInfo[i, 1] = Rl
                        else:
                            self.VInfo[i, 1] = -Rr
                    else:
                        Rl = (Ml + Mr) / L
                        self.VInfo[i, 1] = Rl
        
        self.VInfo = self.VInfo[1:-1, :]

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
        ax.set_title('LINEA DE INFLUENCIA DE CORTANTES')

        x_coords = np.hstack((self.LI_Geom[:, 0],self.LI_Geom[-1,1]))
        y_coords = np.zeros(self.Nnodos)
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
                ax.plot(X_0, -1 * max(abs(self.VInfo[:, 1])), 'ro', markersize=7)
                ax.plot([X_0, X_0], [-1 * max(abs(self.VInfo[:, 1])), 0], color='red', linestyle='--')
                break
        
        max_index = np.argmax(np.abs(self.VInfo[:, 1]))
        max_value = self.VInfo[max_index, 1]
        max_x = self.VInfo[max_index, 0]
        ax.annotate(f'Máx: {max_value:.6f}', (max_x, max_value), xytext=(max_x + 2, max_value),
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