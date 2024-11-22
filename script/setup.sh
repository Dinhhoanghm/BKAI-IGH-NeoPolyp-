# install package
pip install -e .

# curl for dataset, download if not already downloaded
if [ -d "bkai-igh-neopolyp" ]; then
  echo "bkai-igh-neopolyp already downloaded"
else
  echo "bkai-igh-neopolyp not downloaded, downloading now"
  curl -L "https://www.kaggle.com/competitions/30892/download-all" > "bkai-igh-neopolyp.zip"
  unzip -q "bkai-igh-neopolyp.zip" -d "bkai-igh-neopolyp"
fi

