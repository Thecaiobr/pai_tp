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
        self.zoom_mode = False  # Indicates whether the panel is in zoom mode

        self.SetSize(parent.GetSize())

        # Criando a estrutura da imagem
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.imshow_object = None
        self.current_zoom = 1.0

        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Load the image
        self.load_image(filename)


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


    def zoom_in(self, factor=1.1):
        """ Zoom into the image """
        self.current_zoom /= factor
        self.apply_zoom()

    def zoom_out(self, factor=1.1):
        """ Zoom out of the image """
        self.current_zoom *= factor
        self.apply_zoom()

    def apply_zoom(self):
        """ Apply the current zoom level to the image """
        if self.imshow_object is not None:
            ax = self.imshow_object.axes

            # Get the current limits
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

            # Calculate new limits
            x_range = (x_lim[1] - x_lim[0]) * self.current_zoom
            y_range = (y_lim[1] - y_lim[0]) * self.current_zoom

            x_center = (x_lim[1] + x_lim[0]) / 2
            y_center = (y_lim[1] + y_lim[0]) / 2

            # Set new limits
            ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
            ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])  # Keep the original order for y-limits

            self.canvas.draw()
            self.canvas.flush_events()

    def on_image_selected(self, e=None):
        # ... [rest of your code for on_image_selected]
        self.imshow_object = self.axes.imshow(self.data)
        self.current_zoom = 1.0  # Reset zoom level when a new image is loaded

    def reset_zoom(self):
        """ Reset the image view to the original scale """
        if self.imshow_object is not None:
            self.current_zoom = 1.0
            self.axes.set_xlim(0, self.data.shape[1])
            self.axes.set_ylim(self.data.shape[0], 0)
            self.canvas.draw()


    def load_image(self, filename):
        img = Image.open(filename)
        self.data = np.array(img)
        self.imshow_object = self.axes.imshow(self.data)
        self.canvas.draw()

    def toggle_zoom_mode(self):
        """ Toggle the zoom mode on and off """
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            # Change cursor to magnifying glass
            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_MAGNIFIER))
        else:
            # Change cursor back to default
            self.canvas.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

    def on_click(self, event):
        """ Handle the click event for zooming """
        if event.inaxes != self.axes or not self.zoom_mode: 
            return  # Ignore if not in zoom mode or click is outside the axes

        # Coordinates of the clicked point
        x, y = event.xdata, event.ydata

        # Perform zoom operation
        self.current_zoom /= 1.5  # Modify this factor as needed for zoom intensity

        # Calculate new limits
        x_range = (self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) * self.current_zoom
        y_range = (self.axes.get_ylim()[1] - self.axes.get_ylim()[0]) * self.current_zoom

        self.axes.set_xlim([x - x_range / 2, x + x_range / 2])
        self.axes.set_ylim([y - y_range / 2, y + y_range / 2])

        self.canvas.draw()

        # Turn off zoom mode after zooming
        self.toggle_zoom_mode()

    