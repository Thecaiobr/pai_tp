import wx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.widgets import Button
from PIL import Image
from image_panel import ImagePanel

# Interface principal do programa, mostrando os menus
class Interface(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(Interface, self).__init__(*args, **kwargs)
        self.imagePanel = None
        self.init_ui()


    def init_ui(self):
        self.SetSize((800,600))
        self.SetTitle('Visualizador de Células')

        # Criando o menu principal
        self.create_main_menu()
        
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


    def create_image_menu(self):
        viewMenu = wx.Menu()

        # Opções do Menu de visualização
        viewSegmentedImage = viewMenu.Append(wx.NewId(), '&Segmentar Imagem', 'Segmenta a imagem')
        caracterizeNucleus = viewMenu.Append(wx.NewId(), '&Caracterizar Nucleos', 'Caracteriza o nucleo a partir dos descritores de forma')
        classificateImageNucleus = viewMenu.Append(wx.NewId(), '&Classificar Nucleos', 'Classifica os nucleos da imagem')
        zoomInItem = viewMenu.Append(wx.NewId(), '&Zoom In', 'Increase image size')
        zoomOutItem = viewMenu.Append(wx.NewId(), '&Zoom Out', 'Reset image zoom')

        self.menubar.Append(viewMenu, '&Visualização')

        # Eventos das opções do menu
        self.Bind(wx.EVT_MENU, self.on_zoom_out, zoomOutItem)
        self.Bind(wx.EVT_MENU, self.on_zoom_in, zoomInItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_segment_image, viewSegmentedImage)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_caracterize_nucleus_selected, caracterizeNucleus)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_classificate_image_nucleus_selected, classificateImageNucleus)

        self.SetMenuBar(self.menubar)

    # Arquivo > Sair - Sai do programa
    def on_quit(self, e):
        self.Close()

    def on_zoom_in(self, e):
        self.imagePanel.toggle_zoom_mode()


    def on_zoom_out(self, e):
        """ Event handler for the 'Zoom Out' button """
        self.imagePanel.reset_zoom()

    

    # Ao selecionar Arquivo > Abrir...
    def on_open_file(self, e):
        self.dirname = "./"
        dialog = wx.FileDialog(self, "Abrir o arquivo", self.dirname, "", "png and jpg files (*.png;*.jpg;)|*.png;*.jpg;", wx.FC_OPEN)

        if dialog.ShowModal() == wx.ID_OK:
            directory, filename = dialog.GetDirectory(), dialog.GetFilename()
            file = '/'.join((directory, filename))
            
            # Chama a função de mostrar a imagem carregada
            self.load_image(file)

        dialog.Destroy()


    # Carrega a imagem sísmica e mostra no painel
    def load_image(self, filename)->None:
        self.create_main_menu()
        
        # Fecha o painel aberto anteriormente
        if (self.imagePanel):
            self.imagePanel.Destroy()
        
        # Criando o painel da imagem
        self.imagePanel = ImagePanel(self, filename)

        sizer = wx.BoxSizer()
        sizer.Add(self.imagePanel, 1, wx.EXPAND | wx.ALL)
        self.SetSizer(sizer)
        self.Layout()
        self.create_image_menu()


if __name__ == '__main__':
    app = wx.App()
    frame = Interface(None)
    frame.Show()
    app.MainLoop()
    