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
    self.x = int(x)
    self.y = int(y)
    self.z = int(z)
  
  def __add__(self, other):
    return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

  def __sub__(self, other):
    return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)
    

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

# Given two line segments along an axis, identify whether they overlap (coordinates are inclusive of end points)
def check_segment_overlap(min_x1, max_x1, min_x2, max_x2):
  if (max_x1 < min_x2 or max_x2 < min_x1):
    # No overlap detected
    return False, None, None
  else:
    # Found overlap. Return start and end value of overlapping segment
    coord_list = [min_x1, max_x1, min_x2, max_x2]
    coord_list.sort()
    return True, coord_list[1], coord_list[2]

# Check whether there is any overlapping region between cuboids defined by min/max c1 and c2. If yes, return the region of overlap
def intersection_crop(min_c1, max_c1, min_c2, max_c2):
  x_overlap, min_x, max_x = check_segment_overlap(min_c1.x, max_c1.x, min_c2.x, max_c2.x)
  y_overlap, min_y, max_y = check_segment_overlap(min_c1.y, max_c1.y, min_c2.y, max_c2.y)
  z_overlap, min_z, max_z = check_segment_overlap(min_c1.z, max_c1.z, min_c2.z, max_c2.z)
  
  if (x_overlap and y_overlap and z_overlap):
    return True, Coordinate(min_x, min_y, min_z), Coordinate(max_x, max_y, max_z)
  else:
    return False, None, None
  