# Create intermediate, external, and raw data directories if they don't exist
mkdir ../../data/interm
mkdir ../../data/external
mkdir ../../data/raw

# Download a zip file containing filtered data and save it to the raw data directory
wget -O ../../data/raw/filtered_paranmt.zip  https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip

# Unzip the downloaded zip file and place its contents in the raw data directory
unzip ../../data/raw/filtered_paranmt.zip -d ../../data/raw

# Download a toxicity dataset in CSV format and save it to the external data directory
wget -O ../../data/external/toxicity_en.csv https://github.com/surge-ai/toxicity/raw/main/toxicity_en.csv
