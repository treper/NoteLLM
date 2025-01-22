pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
# Using transformers==4.28.1 directly will cause a bug when using from_pretrained() in BLIP2.
# If you would like to use BLIP2 for representations, you might want to check this solution: https://github.com/huggingface/transformers/pull/22564


