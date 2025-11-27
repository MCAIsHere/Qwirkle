# Libraries used:
pip == 23.2.1;
opencv-python == 4.12.0.88;
numpy == 2.2.6;
python == 3.0+
# How to run the script:
There is a variable called *PHOTO_SERIE* *(line 162)* that keeps the serie id for the 21 upcoming photos.
You have to change it manually (that means from 1 to 2 to 3 to 4 and then 5).

The .txt files should appear into a folder called "result". If the folder doesn't exist, you have to create it next to main.py; For reference:

with open(f"result/{PHOTO_SERIE}_{PHOTO_INDEX_STR}.txt", "w") as file: *(line 215)*


Also make sure img = cv.imread('testare/' + str(PHOTO_SERIE) + "_" + PHOTO_INDEX_STR + ".jpg") *(line 168)* is pointing towards to right folder.
Each serie should take around 5-10 seconds to compile (at least from what I tested)


