cd ../src || exit

datasets=('VLCS' 'TerraIncognita' 'CIFAR10CFull' 'CIFAR100CFull')
models=('ViT-B/16' 'RN50')

for dataset in ${datasets[@]}; do
  for model in ${models[@]}; do
    {
      echo "Caching ${model} embeddings on ${dataset}..."
      python cache_emb.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --use_tip \
        --cuda
    }
  done
done