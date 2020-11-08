from bs4 import BeautifulSoup
import os
import pandas as pd

#"""folder_path is path wanted to read"""
folder_path = "C:/Users/김주현/Desktop/20SW_3학년/졸업작품/보안/IOC/ioc"
ioc_info = []


def printResult():
    for i in range(len(ioc_info)):
        print(ioc_info[i])


def makeCSV():
    dataframe = pd.DataFrame(ioc_info)
    name = "ioc_parsedInfo.csv"
    dataframe.to_csv(name)
    print("making CSVfile is success")


def readIOC():
#    """read file name in folder"""
    file_list = os.listdir(folder_path)
    file_length = len(file_list)

    for k in range(file_length):
        source = folder_path + "/" + file_list[k]

        fp = open(source, "r")
        soup = BeautifulSoup(fp, "html.parser")

        item = soup.find_all("indicatoritem")

        for i in range(len(item)):
            one = {'fileNum':k,'document': 'null', 'search': 'null',  'contentType': 'null', 'content': 'null'}

            one.update(document=item[i].context.get('document'))
            one.update(search=item[i].context.get('search'))
            one.update(contentType=item[i].content.get('type'))
            a = item[i].text
            a = a.replace('\n', '')
            one.update(content=a)

            ioc_info.append(one)


def main():
    readIOC()
    printResult()
    makeCSV()

main()

