from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import numpy as np
import wx
from PIL import Image
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.linalg import pinv
from scipy.spatial.distance import mahalanobis
import tensorflow as tf
import matplotlib.patches as mpatches
import csv

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
        self.filename = filename
        self.load_image(filename)

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
        self.data = img
        self.data_array = img_array

    def find_circles(self):
        image = cv2.imread(self.filename)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image_filt = cv2.medianBlur(image_gray, 5)
        circles = cv2.HoughCircles(image_filt, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)
        return circles
    
    def crop_image(self, x, y, gray=False):
        left = x - 50
        right = x + 50
        top = y - 50
        bottom = y + 50
        image = self.data.crop((left, top, right, bottom))
        if gray:
            gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            return gray_image
        return image
    
    def on_segment_image(self, e=None):
        linhas_filtradas = self.find_circles()
                        
        self.characteristics_list = []  # Lista para armazenar os valores
        self.images_list = []
        index = 0
        # Passando por todos os núcleos daquela imagem
        for row in linhas_filtradas[0]:
            cropped_image = self.crop_image(row[0], row[1])
            cropped_image_np = self.crop_image(row[0], row[1], True)
            # cropped_image_np = cv2.imread(cropped_image, cv2.IMREAD_GRAYSCALE)
                
            # Aplicar a limiarização
            centro_y, centro_x = cropped_image_np.shape[0] // 2, cropped_image_np.shape[1] // 2
            blurred_image = cv2.GaussianBlur(cropped_image_np, (5, 5), 0)
            # limiar baseado no ponto central
            valor_pixel_central = np.round(blurred_image[centro_y, centro_x] * 1.37).astype('int')
            _, img_thresholded = cv2.threshold(blurred_image, valor_pixel_central, 255, cv2.THRESH_BINARY)
            
            # Aplicar o detector de bordas Sobel
            sobelx = cv2.Sobel(img_thresholded, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_thresholded, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(sobel.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            centro_x, centro_y = cropped_image_np.shape[1] // 2, cropped_image_np.shape[0] // 2
            dist_minima = float('inf')
            contorno_central = None
            # Encontrando contorno mais próximo do centro
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    dist = np.sqrt((centro_x - cx) ** 2 + (centro_y - cy) ** 2)
                    if dist < dist_minima:
                        dist_minima = dist
                        contorno_central = contour

            area = 0
            perimeter = 0
            excentricity = 0
            compacity = 0
            
            if contorno_central is not None:
                cv2.drawContours(cropped_image_np, [contorno_central], -1, (0, 255, 0), 2)
                area = cv2.contourArea(contorno_central)
                perimeter = cv2.arcLength(contorno_central, True)

                compacity = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0

                circularidade = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

                if len(contorno_central) >= 5:  # Necessário para ajustar uma elipse
                    (x, y), (eixo_menor, eixo_maior), angle = cv2.fitEllipse(contorno_central)
                    excentricity = np.sqrt(1 - (eixo_menor / eixo_maior) ** 2) if eixo_maior != 0 else 0
                else:
                    excentricity = 0

                cv2.circle(cropped_image_np, (centro_x, centro_y), 2, (255, 0, 0), -1)

            
            # Adicione os valores à lista
            self.characteristics_list.append((index, excentricity, area, compacity))
            self.images_list.append(cropped_image)

            width = 1200
            height = 1200
            dim = (width, height)
            resized_img = cv2.resize(cropped_image_np, dim, interpolation = cv2.INTER_AREA)

            subtitle_height = 50
            font = cv2.FONT_HERSHEY_SIMPLEX
            subtitle1 = np.zeros((subtitle_height, resized_img.shape[1]), dtype=np.uint8)
            cv2.putText(subtitle1, f'Area: {round(area)}, Perimetro: {round(perimeter)}, Compacidade: {round(compacity,2)}', (10, 40), font, 1, (255), 2, cv2.LINE_AA)

            # Create second line of subtitle
            subtitle2 = np.zeros((subtitle_height, resized_img.shape[1]), dtype=np.uint8)
            cv2.putText(subtitle2, f'Circularidade: {round(circularidade,2)}, Excentricidade: {round(excentricity,2)}', (10, 40), font, 1, (255), 2, cv2.LINE_AA)

            # f'Área: {area}, Perímetro: {perimeter}, Compacidade: {compacity}, Circularidade: {circularidade}, Excentricidade: {excentricity}'
            img_with_subtitle = np.vstack((resized_img, subtitle1, subtitle2))
            cv2.imshow('Image', img_with_subtitle)

            # Exibir a imagem
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            index += 1
        with open('images_characteristics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write column headers
            writer.writerow(['index', 'excentricity', 'area', 'compacity'])
            # Write data
            writer.writerows(self.characteristics_list)

    def on_classificate_image_nucleus_selected(self, e=None):
        c_model = tf.keras.models.load_model('6_weights.keras', compile=False)
        c_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        b_model = tf.keras.models.load_model('binary_weights.keras', compile=False)
        b_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

        counter = 0

        for image in self.images_list:
            c_model_result, c_model_percentage = self.complete_model(c_model, image)
            b_model_result, b_model_percentage = self.binary_model(b_model, image)
            print(f'Complete model: {c_model_result}, Binary model: {b_model_result}')
            # Assuming bethesda_system is the value to add
            self.characteristics_list = [list(line) for line in self.characteristics_list]
            self.characteristics_list[counter].append(c_model_result)
            self.characteristics_list = [tuple(line) for line in self.characteristics_list]

            width = 1200
            height = 1200
            dim = (width, height)
            resized_img = cv2.resize(np.array(image), dim, interpolation = cv2.INTER_AREA)

            subtitle_height = 50
            font = cv2.FONT_HERSHEY_SIMPLEX

            subtitle1 = np.zeros((subtitle_height, resized_img.shape[1]), dtype=np.uint8)
            cv2.putText(subtitle1, f'Complete model: {c_model_result}, Acuracia: {c_model_percentage}', (10, 40), font, 1, (255), 2, cv2.LINE_AA)
            subtitle1 = subtitle1.reshape(*subtitle1.shape, 1)  # Add an extra dimension to 'subtitle'
            # Convert the grayscale image to a 3-channel image
            subtitle1 = cv2.cvtColor(subtitle1, cv2.COLOR_GRAY2BGR)

            subtitle2 = np.zeros((subtitle_height, resized_img.shape[1]), dtype=np.uint8)
            cv2.putText(subtitle2, f'Binary model: {b_model_result}, Acuracia: {b_model_percentage}', (10, 40), font, 1, (255), 2, cv2.LINE_AA)
            subtitle2 = subtitle2.reshape(*subtitle2.shape, 1)  # Add an extra dimension to 'subtitle'
            # Convert the grayscale image to a 3-channel image
            subtitle2 = cv2.cvtColor(subtitle2, cv2.COLOR_GRAY2BGR)

            # Now you can combine 'resized_img' and 'subtitle1'
            combined = np.vstack((resized_img, subtitle1, subtitle2))

            cv2.imshow('Image', combined)

            # Exibir a imagem
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            counter += 1

    def preprocess_image(self, image):
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    def complete_model(self, loaded_model, image, e=None):
        # Preprocess the image
        img_array = self.preprocess_image(image)

        # Make predictions
        predictions = loaded_model.predict(img_array)
        class_labels = ['asc-h','asc-us','hsil','lsil','Negative for intraepithelial lesion','scc']
        score = tf.nn.softmax(predictions[0])
        return class_labels[tf.argmax(score)], predictions[0,tf.argmax(score)]

    def binary_model(self, loaded_model, image, e=None):
        # Preprocess the image
        img_array = self.preprocess_image(image)

        # Make predictions
        predictions = loaded_model.predict(img_array)
        class_labels = ['Negative for intraepithelial lesion', 'Positive']
        score = tf.nn.softmax(predictions[0])
        return class_labels[tf.argmax(score)], predictions[0,tf.argmax(score)]
    
    def on_scatter_plot(self, e=None):
        # Load the data
        for i in range(len(self.characteristics_list)):
             self.characteristics_list[i] = [round(val, 2) if isinstance(val, float) else val for val in self.characteristics_list[i]]
        data = np.array(self.characteristics_list)
        compacity = data[:, 3]
        compacity = compacity.astype(np.cfloat)
        area = data[:, 2]
        area = area.astype(np.cfloat)
        bethesda_system = data[:, 4]

        # Create a sorted copy of the data for the scatter plot
        scatter_data = sorted(zip(area, compacity, bethesda_system))

        # Extract the sorted area, compacity, and bethesda_system
        area, compacity, bethesda_system = zip(*scatter_data)

        # Map bethesda_system values to colors
        color_dict = {
            'Negative for intraepithelial lesion': 'red',
            'asc-us': 'blue',
            'lsil': 'green',
            'asc-h': 'yellow',
            'hsil': 'purple',
            'scc': 'orange'
        } 
        colors = [color_dict[val] for val in bethesda_system]

        # Create the scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(area, compacity, c=colors) 
        ax.set_xlabel('Area')
        ax.set_ylabel('Compacity')
        ax.set_title('Scatter Plot')

        # Create a custom legend for the colors
        patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
        ax.legend(handles=patches, title="Bethesda System")

        # Display the scatter plot
        plt.show()

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
        self.imshow_object = self.axes.imshow(self.data_array)
        self.current_zoom = 1.0  # Reset zoom level when a new image is loaded

    def reset_zoom(self):
        """ Reset the image view to the original scale """
        if self.imshow_object is not None:
            self.current_zoom = 1.0
            self.axes.set_xlim(0, self.data_array.shape[1])
            self.axes.set_ylim(self.data_array.shape[0], 0)
            self.canvas.draw()


    def load_image(self, filename):
        img = Image.open(filename)
        self.data = img
        self.data_array = np.array(img)
        self.imshow_object = self.axes.imshow(self.data_array)
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

    