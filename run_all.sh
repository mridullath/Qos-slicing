
set -e
cd "$(dirname "$0")"
[ -d data/5Gdataset/data ] || bash get_data.sh
for part in main cond ablate sweep scale; do python src/exp.py $part; done
python src/analyze.py | tee results/analysis_output.txt
