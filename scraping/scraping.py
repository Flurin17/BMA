from serpapi import GoogleSearch
import json
import csv
import requests # to get image from the web
import shutil # to save it locally
from datetime import datetime
import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

listAllPics = []
flower = "Rhododendron"
def write_csv():
    data_file = open('D:/coding/BMA/scraping/test.csv', 'w', newline='', encoding="utf-8")
    csv_writer = csv.writer(data_file)
    
    count = 0
    for data in listAllPics:
        if count == 0:
            header = data.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(data.values())
    
    data_file.close()

def searchStuff(searchTerm):
    start = 0
    for _ in range(8):
        params = {
        "api_key": "9aac5b3edec5d85d0e320b3f05f3bdc0115ca578d72c12b83265238c35cd67dd",
        "engine": "google",
        "q": searchTerm,
        "google_domain": "google.com",
        "ijn": str(start),
        "num": "100",
        "tbm": "isch",
        "tbs": "il:cl"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        pictureResults = results["images_results"]
        for image in pictureResults:
            image["filename"] = get_random_string(12)
            image["time"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            listAllPics.append(image)
            downloadPicture(image)
        
        start = start + 1
        print(start)
    write_csv()

def downloadPicture(image_Infos):
    image_url = image_Infos["original"]
    print(image_url)

    filename = image_Infos["filename"]
    if ".jpg" not in filename:
        filename = filename + ".jpg"

    fileLocation = "D:/coding/BMA/pictures/" + flower + "/" + filename
    # Open the url image, set stream to True, this will return the stream content.
    try:
        r = requests.get(image_url, stream = True, timeout=20,
            headers={
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "de-DE,de;q=0.9,en;q=0.8",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "sec-ch-ua": '"Chromium";v="94", "Google Chrome";v="94", ";Not A Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "Windows",
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
            })

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            # Open a local file with wb ( write binary ) permission.
            with open(fileLocation,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            print('Image sucessfully Downloaded: ',filename)
        else:
            print(r.status_code)
            print('Image Couldn be retreived. Getting Thumbnail')
            image_url = image_Infos["thumbnail"]
            r = requests.get(image_url, stream = True)
            # Check if the image was retrieved successfully
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True
                # Open a local file with wb ( write binary ) permission.
                with open(fileLocation,'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                print('Thumbnail sucessfully Downloaded: ',filename)
    except:
        print("Failed")

searchStuff(flower)