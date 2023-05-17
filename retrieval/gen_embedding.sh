model="the_embedding_model"

python gen_embedding.py --dataset=news_corpus --type=c4_0 --model=${model} --gpuid=1 --batch_size=256 & 

python gen_embedding.py --dataset=news_corpus --type=c4_1 --model=${model} --gpuid=2 --batch_size=256 &

python gen_embedding.py --dataset=news_corpus --type=c4_2 --model=${model} --gpuid=3 --batch_size=256 & 

python gen_embedding.py --dataset=news_corpus --type=c4_3 --model=${model} --gpuid=0 --batch_size=256 

