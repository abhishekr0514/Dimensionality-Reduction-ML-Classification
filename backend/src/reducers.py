from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_pca(n_components=10):
    """
    Returns PCA reducer.
    """
    return PCA(n_components=n_components)


def get_svd(n_components=10):
    """
    Returns Truncated SVD reducer.
    """
    return TruncatedSVD(n_components=n_components)


def get_lda(n_components=1):
    """
    Returns LDA reducer.
    Note: For binary classification, max components = 1
    """
    return LinearDiscriminantAnalysis(n_components=n_components)