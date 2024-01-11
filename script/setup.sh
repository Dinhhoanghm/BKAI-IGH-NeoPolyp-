# install package
pip install -e .

# curl for dataset, download if not already downloaded
if [ -d "bkai-igh-neopolyp" ]; then
  echo "bkai-igh-neopolyp already downloaded"
else
  echo "bkai-igh-neopolyp not downloaded, downloading now"
  curl -L "https://drive.usercontent.google.com/download?id=1bL0LMsJQEmyBmX0DJBOZLFbbE5UAfRKl&export=download&authuser=0&confirm=t&uuid=92a4e8fc-ab4d-4ad5-9f38-e5eea271e194&at=APZUnTXfRj_fx_Y91Y3Iktzp0_LM%3A1704987258377" > "bkai-igh-neopolyp.zip"
  unzip -q "bkai-igh-neopolyp.zip" -d "bkai-igh-neopolyp"
fi

