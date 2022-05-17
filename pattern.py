import numpy as np
import matplotlib.pyplot as pyp

class Checkerboard:
  
  white = np.ones((10, 10), dtype = np.uint8)
  black = np.zeros((2, 2), dtype = np.uint8) #NEED TO ADD VARIABLES FOR RESOLUTION AND TILE
  print(black)

  white[0:2, 0:2] = black
  white[2:4, 2:4] = black 

  chekerBoard = np.tile(white[0:4, 0:4], (5,5)) # FORMULA IS RESOLUTION/(TILE * 2)
  #pyp.xticks([])
  #pyp.yticks([])
  pyp.imshow(chekerBoard, cmap='gray')
  
class CheckerBoard2:
  
  output = 0

  def __init__(self, resolution, tile_size):
    self.resolution = resolution
    self.tile_size = tile_size

  def draw(self):
    white = np.ones((self.resolution, self.resolution), dtype = np.uint8)
    black = np.zeros((self.tile_size, self.tile_size), dtype = np.uint8)

    white[self.tile_size : self.tile_size * 2, self.tile_size : self.tile_size * 2] = black
    white[self.tile_size * 2 : self.tile_size * 4, self.tile_size * 2 : self.tile_size * 4] = black

    checkerBoard = np.tile(white[self.tile_size : self.tile_size * 4, self.tile_size : self.tile_size * 4], (self.tile_size, self.tile_size))

    pyp.imshow(chekerBoard, cmap='gray')
    
class Circle:
    output = 0

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):

      # create coordinate system
      x = np.arange(0, self.resolution)
      y = np.arange(0, self.resolution)
      xx, yy = np.meshgrid(x, y)

      # create a mask with circular shape
      z = np.sqrt((xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2)
      # cut mask to the specified radius
      self.output = z <= self.radius
      return self.output

    def show(self):
      plt.xticks([])
      plt.yticks([])
      plt.imshow(self.output, cmap='gray')
      plt.show()

    circle = Circle(256, 32, (125, 50))
    circle.draw()
    circle.show()
    
class Spectrum:
  res=256
  op=np.zeros([res,res, 3]) # init the array

  #RGB

  op[:,:,0]= np.linspace(0,1,res)

  op[:,:,1]=np.linspace(0,1,res).reshape (256, 1)

  op[:,:,2]= np.linspace(1,0,res)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(op)
