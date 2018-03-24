import numpy as np
from scipy import sparse,ndimage
from scipy import misc
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage import morphology
import keras.backend as K
from skimage import exposure
from sksparse.cholmod import cholesky   #solver
import sys
import skimage.transform  as trans



### Ajacency Matrix
def get_laplacian_4neigh(im):
    '''
    im used for the dimension of the Laplacian Matrix and weights of the edges
      for now not taking into account the long range connections
      '''
      # parameter
    min_i_diff_sq = 0.0001
    big_window_frac = 0.15
    big_window_overlap_frac = 0.95
    long_range_i_thr = 0.1
    sampling_skip = 3
    nsample = 5

    [nrow, ncol, nch] = im.shape
    numnode = nrow * ncol
    ind = np.r_[:numnode]
    ind_mat = ind.reshape((nrow, ncol))

    im_r = im.reshape((-1, nch))

      # first compute the adjacency matrix
    adjmat = sparse.csc_matrix((numnode, numnode), dtype='float32')

      # here the arrays are row major
      # right edges
    right_neigh_ind = ind_mat + 1
    right_neigh_excl = right_neigh_ind[:, :-1]
    ind_mat_excl = ind_mat[:, :-1]

      # want || I(x) - I(y) ||^2
    neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
        - im_r[right_neigh_excl.flatten(), :]

    i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
    right_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

    right_edges = sparse.coo_matrix((right_wt, (ind_mat_excl.flatten(),
                                                  right_neigh_excl.flatten())),
                                      shape=(numnode, numnode)).tocsc()

    right_edges = right_edges.tocsc()
      # add right and left edges
    adjmat = adjmat + right_edges + right_edges.transpose()

      # down edges
    down_neigh_ind = ind_mat + ncol
    down_neigh_excl = down_neigh_ind[:-1, :]
    ind_mat_excl = ind_mat[:-1, :]

    neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
        - im_r[down_neigh_excl.flatten(), :]

    i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
    down_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

    down_edges = sparse.coo_matrix((down_wt, (ind_mat_excl.flatten(),
                                              down_neigh_excl.flatten())),
                                    shape=(numnode, numnode)).tocsc()

    down_edges = down_edges.tocsc()
      # add down and up edges
    adjmat = adjmat + down_edges + down_edges.transpose()
        # So, adjacency matrix done
    degree = adjmat.sum(axis=1)
    degree_mat = sparse.dia_matrix((degree.flatten(), [0]),
                                     shape=(numnode, numnode))

    laplacian = degree_mat - adjmat

    return laplacian
####

###Getting s and reshaping t_out
def get_s(t_out,nrow,ncol):
  #t_out = (t_out-np.min(t_out))/(np.max(t_out)-np.min(t_out))
  t_out = np.reshape(t_out,(nrow*ncol,1))
  s = np.ones((1,nrow*ncol))
  s = s.astype('float32')
  ind = np.where(t_out==0)[0]
  length = ind.shape[0]
  for i in range(length):
    s[0,ind[i]] = 0
  s = s.tolist()[0]
  s = sparse.diags(s,0,format='csc')
  return s,t_out
####


### Solving the matrix linear equation
def lin_sol(t_out,s,l,nrow,ncol,lamda = 0.2):

  try:  
     chol = cholesky(s + lamda * l)
     t_interp_col = chol.solve_A(s*t_out.flatten())
     t_interp = t_interp_col.reshape((nrow, ncol))
     t_interp = np.clip(t_interp, 0.1, 1)
  except:
     return False
     
  return t_interp
####




def smooth_A_guided_filter(img,A,window_a=50,sigma_a=0.05,):
    A1=guided_filter(img[:,:,0],A[:,:,0],window_a,sigma_a)
    A2=guided_filter(img[:,:,1],A[:,:,1],window_a,sigma_a)
    A3=guided_filter(img[:,:,2],A[:,:,2],window_a,sigma_a)
    A=np.stack((A1,A2,A3),axis=-1)

    return A


def linear_smooth(img,T,A,Count,sig1=0.4,sig2=0.2):
    l = get_laplacian_4neigh(img)
    t=T.flatten()
    s=Count.flatten()
    s = sparse.diags(s,0,format='csc')
    t_final = lin_sol(t, s, l, img.shape[0],img.shape[1],sig1)
    A=smooth_A(img,A,l,s,sig2)
    return t_final,A
	

    

def smooth_A(img,A,l,s,sigma):

    At =A[:,:,0].flatten()
    #print"Done s,At"
    A[:,:,0] = lin_sol(At, s, l, img.shape[0],img.shape[1],sigma)

    At =A[:,:,1].flatten()
    #print"Done s,At"
    A[:,:,1] = lin_sol(At, s, l, img.shape[0],img.shape[1],sigma)

    At =A[:,:,2].flatten()
    #print"Done s,At"
    A[:,:,2] = lin_sol(At, s, l, img.shape[0],img.shape[1],sigma)

    return A

