from __future__ import print_function
import numpy as np
from PIL import Image
import sys

# look up the value of Y for the given indices 
# if the indices are out of bounds, return 0
def compute_log_prob_helper(Y, i, j):
  try:
    return Y[i][j]
  except IndexError:
    return 0

def compute_log_prob(X, Y, i, j, w_e, w_s, y_val):
  result = w_e * X[i][j] * y_val

  result += w_s * y_val * compute_log_prob_helper(Y, i-1, j)
  result += w_s * y_val * compute_log_prob_helper(Y, i+1, j)
  result += w_s * y_val * compute_log_prob_helper(Y, i, j-1)
  result += w_s * y_val * compute_log_prob_helper(Y, i, j+1)
  return result

def denoise_image(X, w_e, w_s):
  m, n = np.shape(X)
  # initialize Y same as X
  Y = np.copy(X)
  # optimization
  max_iter = 10*m*n
  for iter in range(max_iter):
    # randomly pick a location
    i = np.random.randint(m)
    j = np.random.randint(n)
    # compute the log probabilities of both values of Y_ij
    log_p_neg = compute_log_prob(X, Y, i, j, w_e, w_s, -1)
    log_p_pos = compute_log_prob(X, Y, i, j, w_e, w_s, 1)
    # assign Y_ij to the value with higher log probability
    if log_p_neg > log_p_pos:
      Y[i][j] = -1
    else:
      Y[i][j] = 1
    if iter % 100000 == 0:
      print ('Completed', iter, 'iterations out of', max_iter)
  return Y

# preprocessing step
def read_image_and_binarize(image_file):
  im = Image.open(image_file).convert("L")
  A = np.asarray(im).astype(int)
  A.flags.writeable = True

  A[A<128] = -1
  A[A>=128] = 1
  return A

def add_noise(orig):
  A = np.copy(orig)
  for i in range(np.size(A, 0)):
    for j in range(np.size(A, 1)):
      r = np.random.rand()
      if r < 0.1:
        A[i][j] = -A[i][j]
  return A

def convert_from_matrix_and_save(M, filename, display=False):
  M[M==-1] = 0
  M[M==1] = 255
  im = Image.fromarray(np.uint8(M))
  if display:
    im.show()
  im.save(filename)

def get_mismatched_percentage(orig_image, denoised_image):
  diff = abs(orig_image - denoised_image) / 2
  return (100.0 * np.sum(diff)) / np.size(orig_image)

def main():
  # read input and arguments
  orig_image = read_image_and_binarize(sys.argv[1])

  if len(sys.argv) > 2:
    try:
      w_e = eval(sys.argv[2])
      w_s = eval(sys.argv[3])
    except:
      print ('Run as: \npython denoise.py <input_image>\npython denoise.py <input_image> <w_e> <w_s>')
      sys.exit()
  else:
    w_e = 8
    w_s = 10

  # add noise
  noisy_image = add_noise(orig_image)

  # use ICM for denoising
  denoised_image = denoise_image(noisy_image, w_e, w_s)

  # print the percentage of mismatched pixels
  print ('Percentage of mismatched pixels: ', get_mismatched_percentage(orig_image, denoised_image))

  convert_from_matrix_and_save(orig_image, 'orig_image.png', display=False)
  convert_from_matrix_and_save(noisy_image, 'noisy_image.png', display=False)
  convert_from_matrix_and_save(denoised_image, 'denoised_image.png', display=False)

if __name__ == '__main__':
  main()

