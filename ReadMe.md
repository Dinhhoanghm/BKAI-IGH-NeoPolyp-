BKAI-IGH NeoPolyp

Student Name: Đõ Đình Hoàng 

Student ID: 20225445

Inferencing guideline

Step 1: 

git clone <your_repo_url>

cd <your_repo_name>

Step 2: we need to download the "model.pth" from Google Drive 

https://drive.google.com/uc?id=11X5lrZV2QAklZ6n9ReQUmb2yyE_eQCuT&export=download&confirm=t&uuid=501d3c0c-6f65-438c-9857-3a70f62ef5b4'




Inferring

!git clone https://github.com/2uanDM/unet-semantic-segmentation.git

!cp /kaggle/working/model.pth /kaggle/working/unet-semantic-segmentation/

!pip install -q segmentation_models_pytorch

!python /kaggle/working/unet-semantic-segmentation/infer.py