import wx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from PIL import Image

class Interface(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(Interface, self).__init__(*args, **kwargs)
        self.imagePanel = None
        self.init_ui()
        # Criando o menu principal
        self.create_main_menu()

    def init_ui(self):
        self.SetSize((800,600))
        self.SetTitle('Visualizador de Células')

    def create_main_menu(self):
        self.menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        
        # Opções do Menu de arquivo
        fileItem = fileMenu.Append(wx.ID_OPEN, '&Abrir...\tCtrl+O')
        closeItem = fileMenu.Append(wx.ID_EXIT, '&Sair')

        # Adicionando as opções no menu principal
        self.menubar.Append(fileMenu, '&Arquivo')

        self.SetMenuBar(self.menubar)

        # Eventos do Menu
        self.Bind(wx.EVT_MENU, self.on_open_file, fileItem)
        self.Bind(wx.EVT_MENU, self.on_quit, closeItem)

    # Arquivo > Sair - Sai do programa
    def on_quit(self, e):
        self.Close()

    # Ao selecionar Arquivo > Abrir...
    def on_open_file(self, e):
        self.dirname = "./"
        dialog = wx.FileDialog(self, "Abrir a imagem", self.dirname, "", "png and jpg files (*.png;*.jpg;)|*.png;*.jpg;", wx.FC_OPEN)

        if dialog.ShowModal() == wx.ID_OK:
            directory, filename = dialog.GetDirectory(), dialog.GetFilename()
            file = '/'.join((directory, filename))
            
            # Chama a função de mostrar a imagem carregada
            self.load_image(file)

        dialog.Destroy()

    def image_panel(self, filename):
        figure = Figure()
        axes = figure.add_subplot(111)
        img = Image.open(filename)
        img_array = np.array(img)
        axes.imshow(img_array)
        return FigureCanvas(self, -1, figure)


    # Carrega a imagem e mostra no painel
    def load_image(self, filename)->None:        
        # Criando o painel da imagem
        imagePanel = self.image_panel(filename)

        self.sizer = wx.BoxSizer()
        self.sizer.Add(imagePanel, 1, wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)
        self.Layout()
        # self.create_image_menu()


if __name__ == '__main__':
    app = wx.App()
    frame = Interface(None)
    frame.Show()
    app.MainLoop()
    
