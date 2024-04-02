import pandas as pd
from scipy.sparse import csr_matrix

def parse_sparse(file, rows, cols):

    # Read the sparse file using Pandas CSV with a delimiter for spaces between values
    # Name the columns for ease of access when converting to a CSR Matrix
    df = pd.read_csv(file, sep=' ', header=None, names=['row', 'col', 'value'])

    # Create a CSR matrix that contains the nonzero values and the corresponding rows and columns, forcing them to be
    # the correct shape based on the config file given to us
    sparse_matrix = csr_matrix((df['value'], (df['row'], df['col'])), shape=(rows, cols))

    # Print the shape of the sparse matrix
    print(f"Sparse Matrix Shape: {sparse_matrix.shape}")

    return sparse_matrix

