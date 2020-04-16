mkdir -p /install/cmu-fg-bg-similarity/logs
cd /ctceu && python -mSimpleHTTPServer > /install/cmu-fg-bg-similarity/logs/data-server.log 2>&1&
cd /install/cmu-fg-bg-similarity/master_server/HTTPServer/ && python SimpleServer.py > /install/cmu-fg-bg-similarity/logs/master-server.log 2>&1&
cd /install/cmu-fg-bg-similarity/Search_HTTPServer/ && python SimpleServer.py full > /install/cmu-fg-bg-similarity/logs/full-service.log 2>&1&
