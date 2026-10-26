Lambda=0.1

seq_lens='1 2 3 4 5 6 7'
pred_lens='1 2 3 4 5 6 7'
model_name="LSTT"
features="M"
breakfast=1
lunch=1
supper=1



if [ ! -d "./logs/LSTT" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LSTT" ]; then
    mkdir ./logs/LSTT
fi


if [ ! -d "./logs/LSTT/lmm" ]; then
    mkdir ./logs/LSTT/lmm
fi

# Loop through Lambda values
for seq_len in $seq_lens
do

    for pred_len in $pred_lens
    do
        echo "Running with seq_len=$seq_len pred_len=$pred_len"
        python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path data.csv \
        --model_id weight \
        --model $model_name \
        --features  $features\
        --image 0\
        --text 1\
        --text_from_img \
        --fusion "NO" \
        --breakfast $breakfast\
        --lunch $lunch\
        --supper $supper\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --Lambda $Lambda \
        --checkpoints "LSTT/lmm" \
        --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LSTT/lmm/$model_name'_'weight_$seq_len'_'$pred_len'_'$features"_"$breakfast$lunch$supper"_l_"$Lambda.log
        echo "Job submitted for pred_len=$pred_len"
    done

    echo "Job submitted for seq_len=$seq_len"
    echo "  "
done
