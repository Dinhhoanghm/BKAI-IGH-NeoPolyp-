# install package
pip install -e .

# curl for dataset, download if not already downloaded
if [ -d "bkai-igh-neopolyp" ]; then
  echo "bkai-igh-neopolyp already downloaded"
else
  echo "bkai-igh-neopolyp not downloaded, downloading now"
  curl -L "https://drive.usercontent.google.com/download?id=1UnWSmQZFiLLhscTEiCGIpsLdBFNXcTlt&export=download&authuser=0&confirm=t&uuid=862dfef3-962b-45da-873e-d1a052f691c6&at=AENtkXY8TXAmYlgKKm6iiT7hu2zD:1732288083551" > "bkai-igh-neopolyp.zip"
  unzip -q "bkai-igh-neopolyp.zip" -d "bkai-igh-neopolyp"
fi

