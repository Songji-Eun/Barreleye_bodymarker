import numpy as np
import cv2



class BIRADs_feature():
    def __init__(self):
        self.shape = None  # LLM model
        self.shape_list = ['Oval' , 'Round', 'Irregular']

        self.orientation = None  # LLM model
        self.orientation_list = ['Parallel' , 'Not Parallel']

        self.margin = None  # LLM model
        self.marign_list = ['Circumscribed' , 'Indistinct', 'Angular', 'Microlobulated', 'Spiculated']

        self.echopattern = None  # LLM model
        self.echopattern_list = ['Anechoic' , 'Hyperechoic', 'Complex cystic and solid', 'Hypoechoic', 'Isoechoic', 'Heterogeneous']

        self.posteriorfeatures = None  # LLM model
        self.posteriorfeatures_list = [ 'No posterior features ', 'Enhancement', 'Shadowing', 'Combined pattern']

        self.calcifications = None  # LLM model
        self.calcifications_list = [ 'Calcifications in a mass', 'Calcifications outside of a mass', 'Intraductal calcifications', 'No calcifications']

        self.position = None  # LLM model
        self.position_list = [ 'Right Lymph node', 'Right nipple', 'Right UIQ', 'Right UOQ', 'Right LIQ', 'Right LOQ'
                               ,'Left Lymph node', 'Left nipple', 'Left UIQ', 'Left UOQ', 'Left LIQ', 'Left LOQ']
