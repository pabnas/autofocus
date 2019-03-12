import numpy as np
import cv2
import matplotlib.pyplot as plt
from camara_class import *
from threading import Timer,Thread,Event


def nothing(x):
    pass
def spectro(array):
    f = np.fft.fft2(array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1 + np.abs(fshift))

    #fig=plt.figure(figsize=(18, 16), dpi= 80)
    #plt.subplot(121)
    #plt.imshow(array, cmap = 'gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return magnitude_spectrum

img = cv2.imread('real.jpeg',0)

from scipy import signal
def mean_filter(img, N):
    kernel = np.ones((N,N))/(N*N)
    return signal.convolve2d(img, kernel, boundary='symm', mode='same')

img_borrosa = mean_filter(img, 77)

def sum(inicio_x,inicio_y,final_x,final_y,width,height,array):
    sumatoria_cuadrado = 0
    #sumatoria del cuadrado de espectro
    for x in range(inicio_y,final_y):
        for y in range(inicio_x, final_x):
            sumatoria_cuadrado = sumatoria_cuadrado + array[x, y]

    sumatoria = 0
    #sumatoria del sector de x
    for x in range(inicio_y,final_y):
        for y in range(inicio_x, width):
            sumatoria = sumatoria+ array[x, y]
    #sumatoria del sector de y
    for x in range(inicio_y,height):
        for y in range(inicio_x, final_x):
            sumatoria = sumatoria+ array[x, y]


    return sumatoria-sumatoria_cuadrado

def borroso(img,size=1):
    array = spectro(img)
    height = np.size(array, 0)
    width = np.size(array, 1)
    #plt.plot(array[int(height / 2), int(width/2):])
    #plt.show()

    x_inicial = int(width / 2)
    x_final = int(width)
    y_inicial = int(height / 2)
    y_final = int(height)
    acumulado_total = sum(x_inicial,y_inicial,x_final,y_final,width,height,array)
    m = (y_final-y_inicial)/(x_final-x_inicial)

    #indice prueba1
    indices = []
    for x in range(x_inicial,x_final,size):
        y = int(m*x)
        acumulado_linea = sum(x_inicial,y_inicial,x,y,width,height,array)
        if acumulado_linea > (acumulado_total/2):
            indices = x
            break

    #acumulado_bajo = 0
    #x_final = int(width * 0.6)
    #for x in range(x_inicial, x_final, size):
        #y = int(m * x)
        #acumulado_bajo = sum(x_inicial,y_inicial,x,y,width,height,array)
        #if acumulado_bajo > (acumulado_total/2):
            #break
    return indices

#print(borroso(img))
#print(borroso(img_borrosa))

class perpetualTimer():
   def __init__(self,t,hFunction):
      self.t=t
      self.hFunction = hFunction
      self.thread = Timer(self.t,self.handle_function)
   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()
   def start(self):
      self.thread.start()
   def cancel(self):
      self.thread.cancel()


indice =0
cam = Camera(camera_file=0)
cv2.namedWindow('Camera')

mejor = 10000
dirreccion = 1
ultimo = 0
focus = 0

while(True):
    frame = cam.get_frame(focus=focus)
    frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    foco = borroso(frame_g)

    if foco < mejor:
        mejor = foco

    if foco > ultimo:
        if mejor < foco:
            dirreccion *= -1

    ultimo = foco
    focus = focus + (0.05 * dirreccion)
    if (focus < -2) | (focus > 2):
        mejor = 10000
        focus = 0

    cv2.putText(frame,"foco = %.2f" % focus + " mejor=" + str(mejor) + " indice=" + str(foco),(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),1,cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


