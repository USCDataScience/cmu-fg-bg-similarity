cd /images && python -mSimpleHTTPServer > /install/cmu-fg-bg-similarity/logs/data-server.log 2>&1&
cd /install/cmu-fg-bg-similarity/segmentation/Caffe_Segmentation/segscripts/data/final_segmentations/ && python -mSimpleHTTPServer 8001 > /install/cmu-fg-bg-similarity/logs/data-server-fg.log 2>&1&
cd /install/cmu-fg-bg-similarity/master_server/HTTPServer/ && python SimpleServer.py > /install/cmu-fg-bg-similarity/logs/master-server.log 2>&1&
cd /install/cmu-fg-bg-similarity/Search_HTTPServer/ && python SimpleServer.py full > /install/cmu-fg-bg-similarity/logs/full-service.log 2>&1&
cd /install/cmu-fg-bg-similarity/Search_HTTPServer/ && python SimpleServer.py fg > /install/cmu-fg-bg-similarity/logs/fg-service.log 2>&1&

