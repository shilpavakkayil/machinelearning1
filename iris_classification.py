import pandas as pd
if __name__=='__main__':
    #url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv("iris_dataset.csv", names=names)
    print(dataset.shape)
    print(dataset.head())
    print(dataset.describe())
