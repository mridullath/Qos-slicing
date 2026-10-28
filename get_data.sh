# Downloads the public Irish 5G production traces (Raca et al., ACM MMSys 2020) used for traffic set D4.
set -e
cd "$(dirname "$0")"
mkdir -p data && cd data
[ -d 5Gdataset ] || git clone --depth 1 https://github.com/uccmisl/5Gdataset
cd 5Gdataset && unzip -q -o 5G-production-dataset.zip -d data
echo "Dataset ready in data/5Gdataset/data"
