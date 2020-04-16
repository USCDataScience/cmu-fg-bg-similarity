export FILENAME="$1"
curl -X POST http://localhost:8888/CounterfeitElectrics_fullImg -H "content-type:text" -d "http://localhost:8000/${FILENAME}"
