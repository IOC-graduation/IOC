import pandas as pd
from pymongo import MongoClient


import warnings
warnings.filterwarnings(action='ignore')

# Associate with the MongoDB instance with 27017 ports on localhost
client = MongoClient('localhost', 27017)

# Create a database called livestock_information
db = client.ioc_information #database
database = db.ioc_info #collection

# Load the data
df = pd.read_csv('C:/Users/YISOEUN/Downloads/Machine-Learning-Project-master/all_file_separate.csv') #데이터 변경

catalog = df.columns #get columnns
record = df.iloc #get record except columns


doc = database.find({})
print(type(doc))

