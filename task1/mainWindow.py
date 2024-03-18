from settingsWindow import SettingsWindow
from staticFunctions import cstFunctions
from Mesh import Mesh
import numpy as np
import math
import json
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QMenuBar,
    QAction
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("APP")
        self.setGeometry(100, 100, 1200, 400)
        
        # create menu
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        
        settingsAction = QAction(self)
        settingsAction.setText("&Settings")
        settingsAction.triggered.connect(self.onSettingsClick)
        
        startAction = QAction(self)
        startAction.setText("&Start")
        startAction.triggered.connect(self.onStartClick)
        
        menuBar.addAction(settingsAction)
        menuBar.addAction(startAction)
        
        # create settings window
        self.settingsWindow = SettingsWindow()
        self.settingsWindow.clicked.connect(self.onSettingsWindowClicked)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # create plot layout
        plot_layout = QGridLayout()
        main_widget.setLayout(plot_layout)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas, 0, 0, 1, 1)
        
        self.ax0 = self.figure.add_subplot(161)
        self.ax1 = self.figure.add_subplot(162)
        self.ax2 = self.figure.add_subplot(163)
        self.ax3 = self.figure.add_subplot(164)
        self.ax4 = self.figure.add_subplot(165)
        self.ax_all = self.figure.add_subplot(166)
        
        self.ax = [self.ax0, self.ax1, self.ax2, self.ax3, self.ax4, self.ax_all]
        
        # parameters
        self.coef_filename = "DEFAULT"
        self.temp_filename = "DEFAULT"
        self.time_filename = "DEFAULT"
        self.outp_filename = "resultValues.json"
        
        
    def onSettingsClick(self):
        self.settingsWindow.show()
    
    def onStartClick(self):
        #def plot(self):
        #    self.figure.clf()
        #    #ax.plot(x, y, 'r.-')
        #    self.canvas.draw_idle()
        
        if self.coef_filename == "DEFAULT":
            self.eps = [0.05, 0.05, 0.1, 0.01, 0.1]
            self.c = [520, 520, 900, 840, 900]
            self.lmbd = [[0, 20, 0, 0, 0], [20, 0, 130, 0, 0], [0, 130, 0, 10.5, 0], [0, 0, 10.5, 0, 119], [0, 0, 0, 119, 0]]
            #self.lmbd = [[0, 20, 0, 0, 0], [0, 0, 130, 0, 0], [0, 0, 0, 10.5, 0], [0, 0, 0, 0, 119], [0, 0, 0, 0, 0]]
        else:
            self.c, self.eps, self.lmbd = cstFunctions.coef_init(self.coef_filename)
            
        self.C0 = 5.67
        self.A = 0.1
        self.model = Mesh("model2.obj")
        self.LS = self.model.s_ij*self.lmbd
        self.EpSC = self.eps*self.model.s_i*self.C0
        #print(self.LS, self.EpSC)
            
        if self.temp_filename == "DEFAULT":
            root = [0, 0, 0, 0, 0]
            self.T = fsolve(cstFunctions.g0, root, args=(self.LS, self.EpSC, self.A, self.c))
        else:
            self.T = cstFunctions.temp_init(self.temp_filename)
        
        if self.time_filename == "DEFAULT":
            self.t_end = 1000
            self.t = np.linspace(1, self.t_end, 301)
        else:
            self.t, self.t_end = cstFunctions.time_init(self.time_filename)
        
        #solve
        self.T = [0, 0, 0, 10, 10]
        #self.T = [300, 30, 30, 30, 300]
        print("Initial temperature", self.T)
        
        self.sol1 = odeint(cstFunctions.g1, self.T, self.t, args=(self.LS, self.EpSC, self.A, self.c))
        #self.sol2 = solve_ivp(cstFunctions.g2, [0, self.t_end], self.T, args=(self.LS, self.EpSC, self.A, self.c), t_eval=self.t, method='Radau')
        #print(self.sol1)
        
        #draw
        
        # for odeint
        for i in range(5):
            self.ax[i].plot(self.t, self.sol1[:, i], label=f"y[{i}]")
            self.ax[5].plot(self.t, self.sol1[:, i], label=f"y[{i}]")
            self.ax[i].legend()
        self.ax[5].legend()
        self.canvas.draw_idle()
        
        # for solve_ivp
        #for i in range(5):
        #    self.ax[i].plot(self.sol2.t, self.sol2.y[i, :], label=f"y[{i}]")
        #    self.ax[5].plot(self.sol2.t, self.sol2.y[i, :], label=f"y[{i}]")
        #    self.ax[i].legend()
        #self.ax[5].legend()
        #self.canvas.draw_idle()
        
        #write to json output file 
        
        # for odeint
        dictionary = {
            "t": self.t.tolist(),
            "y0": self.sol1[:, 0].tolist(),
            "y1": self.sol1[:, 1].tolist(),
            "y2": self.sol1[:, 2].tolist(),
            "y3": self.sol1[:, 3].tolist(),
            "y4": self.sol1[:, 4].tolist()
        }
        json_object = json.dumps(dictionary, indent=6)
        
        # for solve_ivp
        #dictionary = {
        #    "t": self.t.tolist(),
        #    "y0": self.sol2.y[0, :].tolist(),
        #    "y1": self.sol2.y[1, :].tolist(),
        #    "y2": self.sol2.y[2, :].tolist(),
        #    "y3": self.sol2.y[3, :].tolist(),
        #    "y4": self.sol2.y[4, :].tolist()
        #}
        #json_object = json.dumps(dictionary, indent=6)
 
        with open(self.outp_filename, "w") as outfile:
            outfile.write(json_object)

    
    def onSettingsWindowClicked(self):
        if self.settingsWindow.coef_path_radio_default.isChecked():
            self.coef_filename = "DEFAULT"
        else:
            self.coef_filename = self.settingsWindow.coef_path_edit.text()
            
        if self.settingsWindow.temp_path_radio_default.isChecked():
            self.temp_filename = "DEFAULT"
        else:
            self.temp_filename = self.settingsWindow.temp_path_edit.text()
        
        if self.settingsWindow.time_path_radio_default.isChecked():
            self.time_filename = "DEFAULT"
        else:
            self.time_filename = self.settingsWindow.time_path_edit.text()
        
        if self.settingsWindow.outp_path_radio_default.isChecked():
            self.outp_filename = "DEFAULT"
        else:
            self.outp_filename = self.settingsWindow.outp_path_edit.text()
        print(self.coef_filename, self.temp_filename, self.time_filename, self.outp_filename)
