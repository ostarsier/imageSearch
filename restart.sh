#! /bin/bash
fuser -k -n tcp 5000
nohup python server.py > app.log 2>&1 &