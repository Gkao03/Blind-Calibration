import scipy.linalg as linalg
import numpy as np

def convmtx(v, n):
    """Generates a convolution matrix
    
    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's transposed to fit the MATLAB return value.     
    That is, v is assumed to be causal, and zero-valued after N.
    """
    X = linalg.convolution_matrix(v, n)
    return X.T
    
## MATLAB OUTPUT:
# >> h = [1 2 3 2 1];
# >> convmtx(h,7)
# 
# ans =
# 
#      1     2     3     2     1     0     0     0     0     0     0
#      0     1     2     3     2     1     0     0     0     0     0
#      0     0     1     2     3     2     1     0     0     0     0
#      0     0     0     1     2     3     2     1     0     0     0
#      0     0     0     0     1     2     3     2     1     0     0
#      0     0     0     0     0     1     2     3     2     1     0
#      0     0     0     0     0     0     1     2     3     2     1
## PYTHON OUTPUT:
# array([[ 1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.],
