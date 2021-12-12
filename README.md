# Video Metadata Generation

    AI based tool that automated the process of generating metadata of videos based upon visual feature learning and provide video search when a query is provided.

    Projects include multiple aspects:
        1. Image Captioning.
        2. Video Summarizer.
        3. Word vectorization. 


## Setup enviournment

```bash
# Create a Virtual Enviournment "video-metadata"
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Create a env
python3 -m venv video-metadata

# Activate env
source ./video-metadata/bin/activate

# Install Necessary Packages
# git clone repository and extract in video-metadata
cd video-metadata
pip install -r requirements.txt

# Deactivate env
deactivate
```

## Executing script

```bash
mkdir temp
mkdir video_db
mkdir caption_db
# Run the App.py
python3 app.py

# Click on link on terminal or Type "http://127.0.0.1:5000/" in web browser.

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
