mkdir ../../data/interm
mkdir ../../data/external
mkdir ../../data/raw

wget -O ../../data/raw/filtered_paranmt.zip  https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip
unzip ../../data/raw/filtered_paranmt.zip -d ../../data/raw

wget -O ../../data/external/toxicity_en.csv https://github.com/surge-ai/toxicity/raw/main/toxicity_en.csv
