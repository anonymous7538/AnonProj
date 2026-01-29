export CUDA_VISIBLE_DEVICES=0

model_names=("Autoformer" "PatchTST" "FEDformer" "Informer" "LightTS" "Pyraformer" "iTransformer" "TSMixer" "SCINet" "Crossformer" "MSGNet" "TimesNet" "FSNet" "Time-FSNet")
model_names=("Autoformer")


datasets=("exchange_rate" "weather" "electricity" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" )
dims=(8 21 321 7 7 7 7 862)
datas=(custom custom custom ETTh1 ETTh1 ETTm1 ETTm1 custom)
batch_size=32
pred_lens=(180 360 720)

for pred_len in "${pred_lens[@]}"; do
for i in "${!datasets[@]}"; do
    data_path="${datasets[i]}"
    dim="${dims[i]}"
    data="${datas[i]}"

 python -u run.py \
     --task_name MiMe_forcasting \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path ${data_path}/${data_path}.csv \
     --model_id Exchange_96_96 \
     --model MiMe \
     --data $data \
     --features M \
     --seq_len 720 \
     --label_len $(( $pred_len / 2 )) \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 1 \
     --factor 3 \
     --enc_in $dim \
     --dec_in $dim \
     --c_out $dim \
     --des 'Exp' \
     --itr 1 \
     --n_heads 4 \
     --train_epochs 3 \
     --batch_size $batch_size \
     --patch_len 16 \
     --d_model 32 \
     --d_ff 128 \
     --eta_factor 0.001 \
     --result_name "${data_path}_MiMe_${pred_len}" \
     --gin_dir "gin_configs" \
     --model_gins baseline.yaml FFT.yaml triangle.yaml prob.yaml band.yaml

 python -u run.py \
     --task_name long_term_forecast \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path ${data_path}/${data_path}.csv \
     --model_id Exchange_96_96 \
     --model ETSformer \
     --data $data \
     --features M \
     --seq_len 720 \
     --label_len $(( $pred_len / 2 )) \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 2 \
     --factor 3 \
     --enc_in $dim \
     --dec_in $dim \
     --c_out $dim \
     --des 'Exp' \
     --itr 1 \
     --n_heads 4 \
     --train_epochs 3 \
     --batch_size $batch_size \
     --patch_len 16 \
     --d_model 32 \
     --d_ff 128 \
     --result_name "${data_path}_ETSformer_${pred_len}"

 python -u run.py \
     --task_name one_net \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path ${data_path}/${data_path}.csv \
     --model_id Exchange_96_96 \
     --model OneNet \
     --data $data \
     --features M \
     --seq_len 720 \
     --label_len $(( $pred_len / 2 )) \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 1 \
     --factor 3 \
     --enc_in $dim \
     --dec_in $dim \
     --c_out $dim \
     --des 'Exp' \
     --itr 1 \
     --n_heads 4 \
     --train_epochs 3 \
     --batch_size $batch_size \
     --patch_len 16 \
     --d_model 32 \
     --d_ff 128 \
     --result_name "${data_path}_${model_name}_${pred_len}"

for model_name in "${model_names[@]}"; do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ${data_path}/${data_path}.csv \
    --model_id Exchange_96_96 \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len 720 \
    --label_len $(( $pred_len / 2 )) \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $dim \
    --dec_in $dim \
    --c_out $dim \
    --des 'Exp' \
    --itr 1 \
    --n_heads 4 \
    --train_epochs 3 \
    --batch_size $batch_size \
    --patch_len 16 \
    --d_model 1000 \
    --d_ff 128 \
    --result_name "${data_path}_${model_name}_${pred_len}"
done
done
done


