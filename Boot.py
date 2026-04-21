wsl
source ~/unet-env/bin/activate
streamlit run app.py
python train.py
python diag_scr.py
python augment_data.py