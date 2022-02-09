from DatasetManager import DatasetManager

def main():
    myDataset = DatasetManager.DatasetManager()
    datafile = "data/healthcare-dataset-stroke-data.csv"
    myDataset.load_csv(filepath=datafile, cat_text_thresold=5)
    myDataset.set_colmltype_ignore(["id"])
    print(myDataset.get_columnsinfos())

# Only call the greeter when run as
# a script (with python mymodule.py)
if __name__ == '__main__':
    main()

