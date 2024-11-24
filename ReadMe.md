BKAI-IGH NeoPolyp

Student Name: Đõ Đình Hoàng 

Student ID: 20225445

Inferencing guideline

Step 1: 

git clone https://github.com/Dinhhoanghm/BKAI-IGH-NeoPolyp-.git

cd BKAI-IGH-NeoPolyp-

Step 2: We need to download the "model3.pth" from Google Drive 

https://drive.google.com/file/d/1sVGM04BydaGmr1Ybro8UnWhMLVqiZ2sY/view?usp=sharing

Put it in the project root folder


Step 3: Install require libraries

pip install -r requirements.txt

Step 4: Run command

python3 infer.py --image_path  < path to image.jpeg > --checkpoint < path to model3.pth > --output_dir < path to your output> 


(Please check the model is name correct after download. Or you can past the path to the downloaded model)