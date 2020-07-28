
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    """
        https://stackoverflow.com/a/30267328
    """
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode

    def fit(self, X, y=None):
        return self # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    def inverse_transform(self, dframe):
            """
            Transform labels back to original encoding.
            """
            if self.columns is not None:
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx]\
                        .inverse_transform(dframe.loc[:, column].values)
            else:
                self.columns = dframe.iloc[:, :].columns
                for idx, column in enumerate(self.columns):
                    dframe.loc[:, column] = self.all_encoders_[idx]\
                        .inverse_transform(dframe.loc[:, column].values)
            return dframe.loc[:, self.columns].values