import pandas as pd

def parse_labels(file):

    #Test line to make sure the parse_labels function is being reached
    print(f"Parsing labels from {file}")

    #Using Pandas read_CSV to read in the file
    df = pd.read_csv(file, header=None)

    #Checks to make sure labels are one column
    if len(df.columns) == 1:

        #Extracting as a Pandas Series while selecting all the rows and the first column (only column)
        labels = df.iloc[:, 0]

    #Just in case the labels are spread out a bit more
    else:
        labels = df

    return labels