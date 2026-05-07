cd ../src || exit

dataset='CIFAR100CFull'
model='ViT-B/16'  # ('ViT-B/16' 'RN50')

algorithms=('no_adapt' 'latte')

for algorithm in ${algorithms[@]}; do
  {
    echo "Running ${algorithm} on ${model}, ${dataset}..."
    python main.py \
      --dataset "${dataset}" \
      --num_clients_per_domain 3 \
      --model "${model}" \
      --algo "${algorithm}" \
      --seed 0 \
      --cuda
  }
done
