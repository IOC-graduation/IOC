import pandas as pd
from pymongo import MongoClient
import os
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore')


def insertMongoDB():
    # set folder path
    folder_path = "C:/Users/YISOEUN/Desktop/project_data"
    file_list = os.listdir(folder_path)
    file_length = len(file_list)

    # Associate with the MongoDB instance with 27017 ports on localhost
    client = MongoClient('localhost', 27017)

    # Create a database called livestock_information
    db = client.ioc_information  # database
    database = db.ioc_info  # collection

    for k in range(file_length):
        # Load the data (set file path)
        file_name = folder_path + "/" + file_list[k]

        # for j in range(len(file_names)):
        df = pd.read_csv(file_name)
        print(file_name + " call data success")  # checking

        catalog = df.columns  # get columnns
        record = df.iloc  # get record except columns
        print(file_name + " data preprocessing success")  # checking

        for i in range(len(df)):  # make dataframe to dictionary
            information = dict(zip(catalog, record[i]))
            database.insert_one(information)  # insert data to mongoDB
            print(i)  # check

        print(file_name + "completed")  # check


def bringFromDB():
    # Associate with the MongoDB instance with 27017 ports on localhost
    client = MongoClient('localhost', 27017)

    # Create a database called livestock_information
    db = client.ioc_information  # database
    database = db.ioc_info  # collection
    dict_info = database.find()  # find data from mongoDB

    print(type(dict_info[1]))  # data type is dictionary

    # change dictionary to dataframe
    df_info = pd.DataFrame(dict_info)

    print(type(df_info))  # checking



def main():
    bringFromDB()


main()
