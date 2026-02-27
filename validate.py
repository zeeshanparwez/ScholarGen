import os
from dotenv import load_dotenv
load_dotenv("./Config/.env")
path = os.getenv("COURSE_DATA_PATH")

if not path:
    print("Environment variable not set")
elif not os.path.isdir(path):
    print("Not a directory:", path)
else:
    print("Contents of", path)
    for item in os.listdir(path):
        print("-", item)
