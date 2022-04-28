
import numpy as np

#lightweight coreset subsampling

def lightweight(data, m):


    """Reduces (NxD) data matrix from N to Np data points.
    Args:
        data: ndarray of shape [N, D]
        Np: number of data points in the coreset
    Returns:
        coreset: ndarray of shape [Np, D]
        weights: 1darray of shape [Np, 1]
    """
    N = data.shape[0]
    # compute mean
    u = np.mean(data, axis=0)

    # compute proposal distribution
    q = np.linalg.norm(data - u, axis=1)**2
    sum = np.sum(q)
    q = 0.5 * (q/sum + 1.0/N)


    # get sample and fill coreset
    samples = np.random.choice(N, m, replace=False ,p=q)
    coreset = data.iloc[samples]
    weights = 1.0 / (q[samples] * m)
    
    return coreset, weights

# uniform coreset subsampling

def uniform(data,m):
    N = data.shape[0]
    # compute mean

    # get sample and fill coreset
    samples = np.random.choice(N,m, replace=False)
    coreset = data.iloc[samples]
    weights = np.full(m,N/m)
    
    return coreset, weights

def adaptiveSampling(matrixA, matrixC1, c2):
    # compute the residual
    matrixQ, matrixR = np.linalg.qr(matrixC1, mode='reduced')
    matrixQQA = np.dot(matrixQ, np.dot(matrixQ.transpose(), matrixA))
    matrixRes = matrixA - matrixQQA
    # compute the sampling probabilites
    matrixRes = np.square(matrixRes)
    prob = sum(matrixRes)
    prob = prob / sum(prob)
    return np.random.choice(matrixA.shape[0], c2, replace=False, p=prob)

def adaptive(data,m):
    N=data.shape[0]
    N2=int(m/2)
    index1 = np.random.choice(data.shape[0], N2, replace=False)
    data1 = data.iloc[index1]
    index2 = adaptiveSampling(data, data1, m-N2)
    data2 = data.iloc[index2]
    data3 = np.concatenate((data1, data2), 0)
    return data3