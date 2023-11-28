import wx
from control.ImagePanel import ImagePanel

# Interface principal do programa, mostrando os menus
class Interface(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(Interface, self).__init__(*args, **kwargs)
        self.imagePanel = None
        self.init_ui()


    def init_ui(self):
        self.SetSize((800,600))
        self.SetTitle('Visualizador de Imagens Sísmicas')

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
        imageMenu = wx.Menu()
        viewMenu = wx.Menu()

        # Opções do Menu de imagem
        volumeItem = imageMenu.Append(wx.NewId(), '&Alterar volume')
        cmapItem = imageMenu.Append(wx.NewId(), '&Alterar Mapa de Cores')
        contrastItem = imageMenu.Append(wx.NewId(), '&Alterar Contraste')
        similarityItem = imageMenu.Append(wx.NewId(), '&Similaridade')

        # Opções do Menu de visualização
        # view2dItem = viewMenu.Append(wx.NewId(), '&2D', 'Visualiza a imagem sísmica em 2D', wx.ITEM_RADIO)
        view3dItem = viewMenu.Append(wx.NewId(), '&Ver imagem em 3D...', 'Visualiza a imagem sísmica em 3D')
        viewCrossLine = viewMenu.Append(wx.NewId(), '&Crossline', 'Visualiza a imagem sísmica em Crossline', wx.ITEM_RADIO)
        viewInline = viewMenu.Append(wx.NewId(), '&Inline', 'Visualiza a imagem sísmica em Inline', wx.ITEM_RADIO)
        viewTimeslice = viewMenu.Append(wx.NewId(), '&Timeslice', 'Visualiza a imagem sísmica em Timeslice', wx.ITEM_RADIO)

        self.menubar.Append(imageMenu, '&Imagem')
        self.menubar.Append(viewMenu, '&Visualização')

        # Eventos das opções do menu
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_volume, volumeItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_cmap, cmapItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_contrast, contrastItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_similarity, similarityItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_3D_view, view3dItem)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_crossline, viewCrossLine)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_inline, viewInline)
        self.Bind(wx.EVT_MENU, self.imagePanel.on_select_timeslice, viewTimeslice)

        self.SetMenuBar(self.menubar)

    # Arquivo > Sair - Sai do programa
    def on_quit(self, e):
        self.Close()

    # Ao selecionar Arquivo > Abrir...
    def on_open_file(self, e):
        self.dirname = "./"
        dialog = wx.FileDialog(self, "Abrir o arquivo", self.dirname, "", "segy, sgy, h5, npy and xyz files (*.segy;*.sgy;*.h5;*.npy;*.xyz)|*.segy;*.sgy;*.h5;*.npy;*.xyz", wx.FC_OPEN)

        if dialog.ShowModal() == wx.ID_OK:
            directory, filename = dialog.GetDirectory(), dialog.GetFilename()
            file = '/'.join((directory, filename))
            
            # Chama a função de mostrar a imagem carregada
            self.seismic_image(file)

        dialog.Destroy()


    # Carrega a imagem sísmica e mostra no painel
    def seismic_image(self, filename)->None:
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
    