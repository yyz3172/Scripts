# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.

# 通过命令行第一个参数传入版本号，默认值为 v0.13.0rc1
# 用法: ./script.sh [version]
# 示例: ./script.sh v0.14.0
VERSION=${1:-v0.13.0rc1}
export IMAGE=quay.io/ascend/vllm-ascend:${VERSION}

docker run \
    --name vllm-ascend-yyz-${VERSION} \
    --shm-size=64g \
    --net=host \
    --privileged \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/yyz/.cache:/root/.cache \
    -v /root/yyz:/root/yyz \
    -v /root/xrx:/root/xrx \
    -v /root/l00856060:/root/l00856060 \
    -v /root/autodl-tmp:/root/autodl-tmp \
    -v /root/yyz/code/vllm-project/vllm:/vllm-workspace/vllm \
    -v /root/yyz/code/vllm-project/vllm-ascend:/vllm-workspace/vllm-ascend \
    -it $IMAGE bash
