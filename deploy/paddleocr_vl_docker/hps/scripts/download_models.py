from huggingface_hub import snapshot_download

models = [
    "PaddlePaddle/PP-LCNet_x1_0_doc_ori",
    "PaddlePaddle/UVDoc",
    "PaddlePaddle/PP-DocBlockLayout",
    "PaddlePaddle/PP-DocLayout_plus-L",
    "PaddlePaddle/PP-LCNet_x1_0_textline_ori",
    "PaddlePaddle/PP-OCRv5_server_det",
    "PaddlePaddle/PP-OCRv5_server_rec",
    "PaddlePaddle/PP-LCNet_x1_0_table_cls",
    "PaddlePaddle/SLANeXt_wired",
    "PaddlePaddle/SLANet_plus",
    "PaddlePaddle/RT-DETR-L_wired_table_cell_det",
    "PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
    "PaddlePaddle/PP-FormulaNet_plus-L"
]
for model in models:
    snapshot_download(repo_id=model, local_dir=f"/root/.paddlex/official_models/{model.split('/')[-1]}", local_dir_use_symlinks=False)