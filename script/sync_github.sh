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
rm -rf third_party/MegEngine
git submodule add -f git@git-core.megvii-inc.com:brain-sdk/MegBrain.git ./third_party/MegEngine
git add .
git commit -m "Update from github $(date)"
# after create the new sync branch please push it to remote and create a new merge request
# git push --set-upstream origin update-internal