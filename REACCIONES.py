import numpy as np
import matplotlib.pyplot as plt

class LIREACCION:
    def __init__(self, LI_Geom, Input2, Step=0.05):
        self.LI_Geom = LI_Geom
        self.Input2 = Input2
        self.Step = Step
        self.VInfo = None
        self.AreaPositiva = None
        self.AreaNegativa = None
        self.Max = None
        self.calculate()

    def calculate(self):
        LI_Geom = self.LI_Geom
        Input2 = self.Input2
        Step = self.Step

        Nnodos = LI_Geom.shape[0] + 1
        MRigGeneral = np.zeros((Nnodos, Nnodos))
        iterr = np.array([0, 1])
        for ke in range(Nnodos - 1):
            Lenght_e = LI_Geom[ke, 1] - LI_Geom[ke, 0]
            EI_e = LI_Geom[ke, 2] * LI_Geom[ke, 3]
            CteK = (2 * EI_e) / Lenght_e
            ke_e = np.array([[2 * CteK, CteK], [CteK, 2 * CteK]])
            for i in iterr:
                for j in iterr:
                    MRigGeneral[ke + i, ke + j] = ke_e[i, j] + MRigGeneral[ke + i, ke + j]

        Np = int((LI_Geom[Nnodos - 2, 1] - LI_Geom[0, 0]) / Step)
        VInfo = np.zeros((Np + 1, 2))

        for i in range(Np):
            XLoad = i * Step + LI_Geom[0, 0]
            MomEmpo = np.zeros([Nnodos, 1])
            VInfo[i, 0] = XLoad
            for p in range(Nnodos - 1):
                if XLoad > LI_Geom[p, 0] and XLoad < LI_Geom[p, 1]:
                    L = LI_Geom[p, 1] - LI_Geom[p, 0]
                    kp = (XLoad - LI_Geom[p, 0]) / L
                    MomEmpo[p, 0] = -L * kp * (1 - kp) ** 2
                    MomEmpo[p + 1, 0] = L * (kp ** 2) * (1 - kp)
            Giros = np.linalg.solve(MRigGeneral, MomEmpo)
            if Input2 == Nnodos:
                Xreac = LI_Geom[Nnodos - 2, 1]
            else:
                Xreac = LI_Geom[Input2 - 1, 0]
            if Xreac == XLoad:
                VInfo[i, 1] = 1
            else:
                M = np.zeros((Nnodos - 1, 2))
                V = np.zeros((Nnodos - 1, 2))
                for p in range(Nnodos - 1):
                    L = LI_Geom[p, 1] - LI_Geom[p, 0]
                    EI = LI_Geom[p, 2] * LI_Geom[p, 3]
                    Kr = (2 * EI) / L
                    MomEmpo = np.zeros((Nnodos + 1, 1))
                    if XLoad > LI_Geom[p, 0] and XLoad < LI_Geom[p, 1]:
                        kp = (XLoad - LI_Geom[p, 0]) / L
                        MomEmpo[p, 0] = -L * kp * (1 - kp) ** 2
                        MomEmpo[p + 1, 0] = L * (kp ** 2) * (1 - kp)
                    else:
                        MomEmpo[p, 0] = 0
                        MomEmpo[p + 1, 0] = 0
                    M[p, 0] = 2 * Giros[p, 0] * Kr + Giros[p + 1, 0] * Kr - MomEmpo[p, 0]
                    M[p, 1] = Giros[p, 0] * Kr + 2 * Giros[p + 1, 0] * Kr - MomEmpo[p + 1, 0]
                    if abs(MomEmpo[p, 0]) > 0 and abs(MomEmpo[p + 1, 0]) > 0:
                        V[p, 0] = (M[p, 0] + M[p, 1]) / L + (1 - kp)
                        V[p, 1] = 1 - V[p, 0]
                    else:
                        V[p, 0] = (M[p, 0] + M[p, 1]) / L
                        V[p, 1] = -V[p, 0]
                if Input2 == 1:
                    VInfo[i, 1] = V[0, 0]
                else:
                    if Input2 == Nnodos:
                        VInfo[i, 1] = V[Nnodos - 2, 1]
                    else:
                        VInfo[i, 1] = V[Input2 - 1, 0] + V[Input2 - 2, 1]

        self.VInfo = VInfo[1:-1, :]
        self.AreaPositiva = np.sum(self.VInfo[self.VInfo[:, 1] > 0, 1]) * Step
        self.AreaNegativa = np.sum(self.VInfo[self.VInfo[:, 1] < 0, 1]) * Step
        self.Max = np.max(np.abs(self.VInfo[:, 1]))

    def plot(self):
        VInfo = self.VInfo
        LI_Geom = self.LI_Geom
        Input2 = self.Input2
        Step = self.Step

        plt.style.use('classic')
        fig, ax = plt.subplots()
        ax.plot(VInfo[:, 0], VInfo[:, 1])

        ax.fill_between(VInfo[:, 0], VInfo[:, 1], 0, where=(VInfo[:, 1] < 0), color='lightcoral', alpha=0.3)
        ax.fill_between(VInfo[:, 0], VInfo[:, 1], 0, where=(VInfo[:, 1] >= 0), color='skyblue', alpha=0.3)

        ax.set_xlabel('Distancia del puente')
        ax.set_ylabel('Valor en la línea de influencia')
        ax.set_title('Linea de influencia')

        x_coords = np.hstack((LI_Geom[:, 0], LI_Geom[-1, 1]))
        y_coords = np.zeros(len(x_coords))
        ax.plot(x_coords, y_coords, color='#000000', linewidth=1.5)

        if Input2 == len(LI_Geom) + 1:
            ax.plot(LI_Geom[-1, 1], 1, 'ro', markersize=7)
            ax.plot([LI_Geom[-1, 1], LI_Geom[-1, 1]], [1, 0], color='red', linestyle='--')
        else:
            ax.plot(LI_Geom[Input2 - 1, 0], 1, 'ro', markersize=7)
            ax.plot([LI_Geom[Input2 - 1, 0], LI_Geom[Input2 - 1, 0]], [1, 0], color='red', linestyle='--')

        Max = self.Max
        max_index = np.argmax(np.abs(VInfo[:, 1]))
        max_value = VInfo[max_index, 1]
        max_x = VInfo[max_index, 0]

        ax.annotate(f'Máx: {max_value:.2f}', (max_x, max_value), xytext=(max_x +0.5, max_value),
                    arrowprops=dict(facecolor='black', arrowstyle='-'))

        markers = ['s'] + ['^'] * (len(x_coords) - 1)
        colors = ['#000000'] * len(x_coords)
        for x, y, marker, color in zip(x_coords, y_coords, markers, colors):
            ax.scatter(x, y, color=color, marker=marker, s=60)
        ax.grid(True)
        return fig

    def MaxValue(self):
        return self.Max

    def AreaPositiva(self):
        return self.AreaPositiva

    def AreaNegativa(self):
        return self.AreaNegativa
    
    
