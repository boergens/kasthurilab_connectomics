'''
@File    :   coordinate.py
@Time    :   2023/06/28 12:37:52
@Author  :   Surya Chandra Kalia
@Version :   1.0
@Contact :   suryackalia@gmail.com
@Org     :   Kasthuri Lab, University of Chicago
'''

class Coordinate:
  def __init__ (self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    

def minCoord(c1, c2):
  return Coordinate(min(c1.x, c2.x), min(c1.y, c2.y), min(c1.z, c2.z))

def maxCoord(c1, c2):
  return Coordinate(max(c1.x, c2.x), max(c1.y, c2.y), max(c1.z, c2.z))

# Give a lower bound to minCoord upon dilation by given amount but restricted by the image shape
def dilateMin(coord, dilation_amt, img_shape):
  return Coordinate(max(coord.x - dilation_amt, 0), max(coord.y - dilation_amt, 0), max(coord.z - dilation_amt, 0))

# Give a upper bound to minCoord upon dilation by given amount but restricted by the image shape. Here img_shape = (z, y, x)
def dilateMax(coord, dilation_amt, img_shape):
  return Coordinate(min(coord.x + dilation_amt, img_shape[2]-1), min(coord.y + dilation_amt, img_shape[1]-1), min(coord.z + dilation_amt, img_shape[0]-1))