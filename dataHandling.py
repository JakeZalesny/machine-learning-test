import pandas as pd
import numpy as nump
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier

class DataHandling: 
    def __init__(self, data:str) -> None:
        self.data = data
        self.df = pd.read_csv(self.data)
        self.array = self.df.values
        self.model = DecisionTreeClassifier() 
        self.X = self.df.drop(columns=self.df.loc["Rank", "Name", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"])
        self.Y = self.df["Global_Sales"]
    
    def display_raw_data(self) -> None: 
        print(self.df)
        print(self.df.describe())
    
    def display_data_array(self) -> None:
        print(self.array)
    
    def make_decision_tree(self) -> None: 
        self.model.fit(self.X, self.Y)
    
    def predict_global_sales(self): 
        predictions = self.model.predict([ [2006, "Wii"], [1995, "NES"], [1999, "GBA"] ])
        return predictions

