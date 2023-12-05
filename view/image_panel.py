from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import numpy as np
import wx
from PIL import Image

class ImagePanel(wx.Panel):
    def __init__(self, parent, filename):
        super(ImagePanel, self).__init__(parent)

        # Definindo as variáveis que serão utilizadas
        self.volume = 0
        self.volumeInfo = None
        self.cmap = 'Greys'
        self.cmapSelect = None
        self.contrast = 2
        self.contrastInfo = None

        self.SetSize(parent.GetSize())

        # Criando a estrutura da imagem
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Executando o cube do segyio em outra thread
        print("~ Criando a imagem...")
        thread = threading.Thread(target=self.get_data, args=(filename,))
        thread.start()
        print('Carregando...')
        thread.join()
        self.on_image_selected()
        print("~ Imagem criada!")

        # Sizer para o conteudo do painel
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL)
        self.sizer.Fit(self)
        self.SetSizer(self.sizer)

    def get_data(self, filename):
        img = Image.open(filename)
        img_array = np.array(img)
        self.data = img_array

    def on_image_selected(self, e=None):
        self.sim = self.axes.imshow(self.data)

    def on_caracterize_nucleus_selected(self, e=None):
        pass

    def on_classificate_image_nucleus_selected(self, e=None):
        pass
