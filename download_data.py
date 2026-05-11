from src.data import DataGetter


if __name__ == "__main__":
    ptbdb_data = DataGetter(dataset="ptbdb")
    ptbdb_data.run()

    ptbxl_data = DataGetter(dataset="ptb-xl")
    ptbxl_data.run()
    
    ludb_data = DataGetter(dataset="ludb")
    ludb_data.run()
