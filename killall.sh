nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
kill -9 $(lsof -t -i:8787)
