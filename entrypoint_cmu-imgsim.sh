#!/bin/bash

$DEPLOY_HOME=/install/cmu-fg-bg-similarity/
$SCRIPTS_HOME=$DEPLOY_HOME/scripts
$SEGMENTATION_HOME=$DEPLOY_HOME/segmentation/Caffe_Segmentation/


echo "Docker Container ID:" $HOSTNAME

$SCRIPTS_HOME/clean-paths.sh
$SCRIPTS_HOME/gen-file-list.sh
pushd $SCRIPTS_HOME
./run_FeatExt_full.sh
popd

pushd $SCRIPTS_HOME
./buildIndex_ITQ.sh
popd

pushd $SCRIPTS_HOME
./buildIndex_ITQ.sh
popd

pushd $SCRIPTS_HOME
./run_server_ITQ.sh
popd

pushd $SEGMENTATION_HOME/data
./gen-segmentation-input.sh
popd

pushd $SEGMENTATION_HOME/scripts
./run_seg.sh ..
popd

pushd $SCRIPTS_HOME
./run_FeatExt_fg.sh
popd

pushd $SCRIPTS_HOME
./buildIndex_fg_ITQ.sh
popd

pushd $SCRIPTS_HOME
./run_server_fg_ITQ.sh
popd

pushd $SCRIPTS_HOME
./web_services.sh
popd

# watch log
echo "Watching logs...."
tail -f $DEPLOY_HOME/logs/*.log
