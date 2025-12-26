conda create -n ultrashape python=3.10
conda activate ultrashape 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
