# /bin/bash
set -x

git fetch 
git checkout github-main
git remote add external git@github.com:MegEngine/MegCC.git
git fetch external

export PATCH_COMMIT="$(git rev-parse HEAD)"
git branch --create update-internal 
git checkout update-internal
git reset --hard external/main
git cherry-pick -X ours $PATCH_COMMIT  
git am patchs/*
git reset origin/dev
rm -rf patchs 
git add .
git commit -m "Update from github $(date)"
echo "after create the new sync branch please push it to remote and create a new merge request: \"git push --set-upstream origin update-internal\""