import numpy as np


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    output = []
    for i in range(x.shape[0]):
        mu = np.mean(x[i])
        output.append(x[i]-mu)
    output = np.array(output)
    return output


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    output = []
    for i in range(x.shape[0]):
        sigma = np.var(x[i])
        output.append(scale * x[i] / np.sqrt(bias + sigma))
    output = np.array(output)
    return output


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    x_output = np.copy(x)
    xtest_output = np.copy(xtest)
    mu = np.mean(x, axis=0)
    x_output = np.subtract(x_output, mu)
    xtest_output = np.subtract(xtest_output, mu)
    return x_output, xtest_output
        


def zca(x, xtest, bias=1e-4):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    eye = np.eye(x.shape[1])
    U, S, _ = np.linalg.svd(np.dot(x.T, x) / x.shape[0] + eye * bias)
    pca = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    zca_x = np.dot(x, pca)
    zca_xtest = np.dot(xtest, pca)
    
    return zca_x, zca_xtest


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    gcn_x = gcn(sample_zero_mean(x))
    gcn_xtest = gcn(sample_zero_mean(xtest))
    fzm_x, fzm_xtest = feature_zero_mean(gcn_x, gcn_xtest)
    zca_x, zca_xtest = zca(fzm_x, fzm_xtest)
    return zca_x.reshape(x.shape[0], 3, image_size, image_size), zca_xtest.reshape(xtest.shape[0], 3, image_size, image_size)


