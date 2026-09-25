export CUDA_VISIBLE_DEVICES=0

model_names=("FEDformer" "Pyraformer" "Crossformer" "iTransformer" "PatchTST" "LightTS" "TSMixer" 'TimeMixer' 'WPMixer' 'PAttn' "TimesNet" "FSNet" "Time-FSNet")

datasets=("exchange_rate" "weather" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" )
dims=(8 21 7 7 7 7 862)
datas=(custom custom ETTh1 ETTh1 ETTm1 ETTm1 custom)
batch_size=32
pred_lens=(12 24)

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


