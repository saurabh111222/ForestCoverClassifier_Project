
class selectSchema:
    def __init__(self):
        pass
    def predSchema(self):
            schemaForPrediction = ["elevation", "aspect", "slope", "horizontal_distance_to_hydrology",
                               "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
                               "Horizontal_Distance_To_Fire_Points", "wilderness_area1", "wilderness_area2",
                               "wilderness_area3", "wilderness_area4", "soil_type_1", "soil_type_2", "soil_type_3",
                               "soil_type_4", "soil_type_5", "soil_type_6", "soil_type_7", "soil_type_8", "soil_type_9",
                               "soil_type_10", "soil_type_11", "soil_type_12", "soil_type_13", "soil_type_14",
                               "soil_type_15", "soil_type_16", "soil_type_17", "soil_type_18", "soil_type_19",
                               "soil_type_20", "soil_type_21", "soil_type_22", "soil_type_23", "soil_type_24",
                               "soil_type_25", "soil_type_26", "soil_type_27", "soil_type_28", "soil_type_29",
                               "soil_type_30", "soil_type_31", "soil_type_32", "soil_type_33", "soil_type_34",
                               "soil_type_35", "soil_type_36", "soil_type_37", "soil_type_38", "soil_type_39",
                               "soil_type_40"]
            return schemaForPrediction
    def trainingSchema(self):
            schemaForTraining = ["elevation", "aspect", "slope", "horizontal_distance_to_hydrology",
                                   "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
                                   "Horizontal_Distance_To_Fire_Points", "wilderness_area1", "wilderness_area2",
                                   "wilderness_area3", "wilderness_area4", "soil_type_1", "soil_type_2", "soil_type_3",
                                   "soil_type_4", "soil_type_5", "soil_type_6", "soil_type_7", "soil_type_8",
                                   "soil_type_9","soil_type_10", "soil_type_11", "soil_type_12", "soil_type_13", "soil_type_14",
                                   "soil_type_15", "soil_type_16", "soil_type_17", "soil_type_18", "soil_type_19",
                                   "soil_type_20", "soil_type_21", "soil_type_22", "soil_type_23", "soil_type_24",
                                   "soil_type_25", "soil_type_26", "soil_type_27", "soil_type_28", "soil_type_29",
                                   "soil_type_30", "soil_type_31", "soil_type_32", "soil_type_33", "soil_type_34",
                                   "soil_type_35", "soil_type_36", "soil_type_37", "soil_type_38", "soil_type_39",
                                   "soil_type_40",'class']
            return schemaForTraining
