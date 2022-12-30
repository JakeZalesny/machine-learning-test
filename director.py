import pandas as pd
import numpy as nump
import matplotlib as mpl
import sklearn as sk 
from dataHandling import DataHandling

class Director: 
    def __init__(self) -> None:
        pass

    def direct(self):
        dh = DataHandling("vgsales.csv")
        dh.display_raw_data()
        dh.make_decision_tree()
        predictions = dh.predict_global_sales()
        print(predictions)