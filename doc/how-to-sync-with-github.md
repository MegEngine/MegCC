# How to sync GitHub code into internal Gitlab repo

## get the sync base branch origin/github-main

```bash
    git fetch
    git checkout github-main 
    git reset --hard origin/github-main
```

## run the sync script

```bash
    bash ./script/sync_github.sh
```
:warning: **If there is conflict and other problems**: please run the cmdline one by one in [sync_github.sh](../script/sync_github.sh) script and fix all problems 

## 