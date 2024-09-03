find . -name '*.pyc' 
find . -name '*.pyc' | xargs -I {} rm -rf {}
find . -name '*__pycache__*'
find . -name '*__pycache__*' | xargs -I {} rm -rf {}
rm -rf ./vis/
rm -rf ./runs/