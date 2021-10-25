from serpapi import GoogleSearch
import json
import csv
import requests # to get image from the web
import shutil # to save it locally


listAllPics = []

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


def searchStuff(searchTerm):
    start = 0
    for _ in range(3):
        params = {
        "api_key": "9aac5b3edec5d85d0e320b3f05f3bdc0115ca578d72c12b83265238c35cd67dd",
        "engine": "google",
        "q": searchTerm,
        "location": "Switzerland",
        "google_domain": "google.ch",
        "gl": "ch",
        "hl": "de",
        "start": str(start),
        "num": "100",
        "tbm": "isch",
        "no_cache": "true"
        }
    start += 100
    search = GoogleSearch(params)
    results = search.get_dict()
    pictureResults = results["images_results"]
    listAllPics.append(pictureResults)



def downloadPicture(image_Infos):
    image_url = image_Infos["original"]

    filename = image_Infos["title"]

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived. Getting Thumbnail')
        image_url = image_Infos["thumbnail"]