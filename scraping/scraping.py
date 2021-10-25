from serpapi import GoogleSearch
 
import json
import csv

def write_csv(jsonData):
    data_file = open('D:/coding/BMA/scraping/test.csv', 'w', newline='')
    csv_writer = csv.writer(data_file)
    
    count = 0
    for data in jsonData:
        if count == 0:
            header = data.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(data.values())
    
    data_file.close()